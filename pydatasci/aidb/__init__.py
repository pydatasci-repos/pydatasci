name = "aidb"

import os, sqlite3, io, gzip, zlib
from datetime import datetime

#orm
from peewee import *
from playhouse.dataset import DataSet
from playhouse.sqlite_ext import SqliteExtDatabase, JSONField
#etl
import pyarrow
from pyarrow import parquet as pq
from pyarrow import csv as pc
import pandas as pd
import numpy as np

# Assumes `pds.create_config()` is run prior to `pds.get_config()`.
from pydatasci import get_config


def get_path_db():
	pds_config = get_config()
	if pds_config is None:
		# pds will provide an explanatory print() statement.
		pass
	else:
		db_path = pds_config['db_path']
		return db_path


def get_db():
	path = get_path_db()
	if path is None:
		print("\n Error - Cannot fetch database because it has not yet been configured.\n")
	else:
		# peewee ORM connection to database:
		#db = SqliteDatabase(path)
		db = SqliteExtDatabase(path)
		return db


def get_path_dataset_ext():
	path = get_path_db()
	if path is None:
		print("\n Error - Cannot fetch database because it has not yet been configured.\n")
	else:
		# Despite looking weird, this path works on Windows: `sqlite:///C:\\...`
		prefix = "sqlite:///"
		prefix_db_path = prefix + path
		return prefix_db_path


def get_dataset_ext():
	# http://docs.peewee-orm.com/en/latest/peewee/playhouse.html#dataset
	path = get_path_dataset_ext()
	db = DataSet(path)
	return db


def create_db():
	# Future: Could let the user specify their own db name, for import tutorials. Could check if passed as an argument to create_config?
	db_path = get_path_db()
	db_exists = os.path.exists(db_path)
	if db_exists:
		print("\n=> Skipping database file creation as a database file already exists at path:\n" + db_path + "\n")
	else:
		# Create sqlite file for db.
		try:
			db = get_db()
		except:
			print("\n=> Error - failed to create database file at path:\n" + db_path)
			print("===================================\n")
			raise
		print("\n=> Success - created database file for machine learning metrics at path:\n" + db_path + "\n")

	db = get_db()
	# Create tables inside db.
	tables = db.get_tables()
	table_count = len(tables)
	if table_count > 0:
		print("\n=> Info - skipping table creation as the following tables already exist:\n" + str(tables) + "\n")
	else:
		db.create_tables([Job, Dataset])
		tables = db.get_tables()
		table_count = len(tables)
		if table_count > 0:
			print("\n=> Success - created the following tables within database:\n" + str(tables) + "\n")
		else:
			print("\n=> Error - failed to create tables. Please see README file section titled: 'Deleting & Recreating the Database'\n")


def delete_db(confirm:bool=False):
	if confirm:
		db_path = get_path_db()
		db_exists = os.path.exists(db_path)
		if db_exists:
			try:
				os.remove(db_path)
			except:
				print("\n=> Error - failed to delete database file at path:\n" + db_path)
				print("===================================")
				raise
			print("\n=> Success - deleted database file at path:\n" + db_path + "\n")

		else:
			print("\n=> Info - there is no file to delete at path:\n" + db_path + "\n")
	else:
		print("\n=> Info - skipping deletion because `confirm` arg not set to boolean `True`.\n")



# ============ CRUD ============
def create_dataset_from_file(
	path:str,
	file_format:str,
	name:str=None,
	perform_gzip:bool=True
):
	"""
	- File is read in with pyarrow, converted to bytes, compressed by default, and stored as a SQLite blob field.
	- Note: If you do not remove your file's index columns before importing them, then they will be included in your Dataset. The ordered nature of this column represents potential bias during analysis. You can drop these and other columns in memory when creating a Featureset from your Dataset.
	- Note: If no column names are provided, then they will be inserted automatically.
	- `path`: Local or absolute path
	- `file_format`: Accepts uncompressed formats including parquet, csv, and tsv (a csv with `delimiter='\t'`). This tag is used to tell pyarrow how to handle the file. We do not infer the path because (a) we don't want to force file extensions, (b) we want to make sure users know what file formats we support.
	- `name`: if none specified, then `path` string will be used.
	- `perform_gzip`: Whether or not to perform gzip compression on the file. We have observed up to 90% compression rates during testing.
	"""
	
	# create some files with no column names
	# do some testing with sparse null column names...
	# do some testing with all null column names...
	accepted_formats = ['csv', 'tsv', 'parquet']
	if file_format not in accepted_formats:
		print("Error - Accepted file formats include uncompressed csv, tsv, and parquet.")
	else:
		if name is None:
			name=path

		if perform_gzip is None:
			perform_gzip=True

		if file_format == 'csv':
			parse_opt = pc.ParseOptions(delimiter=',')
			tbl = pc.read_csv(path)
		elif file_format == 'tsv':
			parse_opt = pc.ParseOptions(delimiter='\t')
			tbl = pc.read_csv(path, parse_options=parse_opt)
		elif file_format == 'parquet':
			tbl = pq.read_table(path)

		#ToDo - handle columns with no name.
		column_names = tbl.column_names

		# should doooo something with it first to normalize it.
		with open(path, "rb") as f:
			bytesio = io.BytesIO(f.read())
			data = bytesio.getvalue()
			if perform_gzip:
				data = gzip.compress(data)
				is_compressed=True
			else:
				is_compressed=False

		d = Dataset.create(
			name = name,
			data = data,
			file_format = file_format,
			is_compressed = is_compressed,
			column_names = column_names
		)
		return d

#def create_dataset_from_pandas():
	#read as arrow
	#save as parquet from some kind of buffer?

#def create_dataset_from_numpy():
	#read as arrow
	#save as parquet from some kind of buffer?

def get_dataset(id:int):
	d = Dataset.get_by_id(id)
	return d


def read_dataset_to_pandas(id:int):
	"""
	- After unzipping `gzip.open()`, bytesio still needed to be read into PyArrow before being read into Pandas.
	"""
	d = get_dataset(id)

	is_compressed = d.is_compressed
	ff = d.file_format
	
	data = d.data
	bytesio_data = io.BytesIO(data)
	if (ff == 'csv') or (ff == 'tsv'):
		if is_compressed:
			bytesio_csv = gzip.open(bytesio_data)
			if ff == 'tsv':
				parse_opt = pc.ParseOptions(delimiter='\t')
				tbl = pc.read_csv(bytesio_csv, parse_options=parse_opt)
			else:
				tbl = pc.read_csv(bytesio_csv)
			df = tbl.to_pandas()
		else:
			if ff == 'tsv':
				df = pd.read_csv(bytesio_data, sep='\t')
			else:
				df = pd.read_csv(bytesio_data)
	elif ff == 'parquet':
		if is_compressed:
			bytesio_parquet = gzip.open(bytesio_data)
			tbl = pq.read_table(bytesio_parquet)
			df = tbl.to_pandas()
		else:
			df = pd.read_parquet(bytesio_data)
	return df


def read_dataset_to_numpy(id:int):
	"""
	ToDo
	- Returns a NumPy structured array: https://numpy.org/doc/stable/user/basics.rec.html
	- There doesn't seem to be a direct Parquet to NumPy, so have to convert through PyArrow or Pandas.
	- Started down this path, but just switched to Pandas: `np.genfromtxt(bytesio_data, names=True, delimiter=',')`. 
	"""
	df = read_dataset_to_pandas(id)
	arr = df.to_records(index=False)
	return arr

#Future: or will np suffice?
#def read_dataset_as_pytorch_tensor():

#read featureset... fetch columns 

# ============ ORM ============
# http://docs.peewee-orm.com/en/latest/peewee/models.html
class BaseModel(Model):
	class Meta:
		database = get_db()


class Job(BaseModel):
	status = CharField()


class Dataset(BaseModel):
	name = CharField()
	data = BlobField()
	file_format = CharField()
	is_compressed = BooleanField()
	column_names= JSONField()
	#compression = CharField()

# remember, Featureset is just columns to use from a Dataset.
