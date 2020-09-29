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
		db.create_tables([Job, Dataset, Label, Supervisedset, Unsupervisedset])
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
	category = CharField() #tabular, image, longitudinal
	file_format = CharField()
	is_compressed = BooleanField()
	column_names= JSONField()
	#compression = CharField()

	def create_from_file(
		path:str
		,file_format:str
		,name:str=None
		,perform_gzip:bool=True
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
			# Defaults.
			if name is None:
				name=path
			if perform_gzip is None:
				perform_gzip=True

			# File formats.
			if file_format == 'csv':
				parse_opt = pc.ParseOptions(delimiter=',')
				tbl = pc.read_csv(path)
			elif file_format == 'tsv':
				parse_opt = pc.ParseOptions(delimiter='\t')
				tbl = pc.read_csv(path, parse_options=parse_opt)
			elif file_format == 'parquet':
				tbl = pq.read_table(path)

			# Category
			category = 'tabular'

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
				name = name
				,data = data
				,file_format = file_format
				,is_compressed = is_compressed
				,column_names = column_names
				,category = category
			)
			return d


	def read_to_pandas(id:int):
		"""
		- After unzipping `gzip.open()`, bytesio still needed to be read into PyArrow before being read into Pandas.
		"""
		d = Dataset.get_by_id(id)

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


	def read_to_numpy(id:int):
		"""
		- Returns a NumPy structured array: https://numpy.org/doc/stable/user/basics.rec.html
		- Started implementing `np.genfromtxt(bytesio_data, names=True, delimiter=',')`, but just switched to Pandas.
		- There doesn't seem to be a direct Parquet to NumPy, so have to convert through PyArrow or Pandas.
		"""
		df = Dataset.read_to_pandas(id)
		arr = df.to_records(index=False)
		return arr

	"""
	Future:
	- Read as Tensors (pytorch and tf)? Or will numpy suffice?
	- Longitudinal data?
	- Images?
	"""

	"""
	ToDo
	- read featureset... fetch columns
	- check that columns are actually named.

	#def create_dataset_from_pandas():
		#read as arrow
		#save as parquet from some kind of buffer?

	#def create_dataset_from_numpy():
		#read as arrow
		#save as parquet from some kind of buffer?
	"""

class Label(BaseModel):
	column_name=CharField()
	
	dataset = ForeignKeyField(Dataset, backref='labels')
	
	def create_from_dataset(
		dataset_id:int
		,column_name:str
	):
		d = Dataset.get_by_id(dataset_id)

		# verify that the column exists
		d_columns = d.column_names
		column_found = column_name in d_columns
		if column_found:
			l = Label.create(
				dataset=d
				,column_name=column_name
			)
			return l
		else:
			print("Error - Column name not found in `Dataset.column_names`.")
			return None


class Featureset(BaseModel):
	"""
	- Unsupervised featuresets do not have a label ForeignKey, whereas Supervised featuresets do, which results in NULL ForeignKey errors.
	^ As a workaround, I originally got `DeferredForeignKey('Label',null=True)` with `tags=[supervised,unsupervised]` working.
	^ However, `.label` only returned an integer not object, `backref` didn't work, and subclass models are so simple.
	"""
	column_names = JSONField()
	tags = JSONField()
	
	dataset = ForeignKeyField(Dataset, backref='featuresets')


class Supervisedset(Featureset):
	# Inherits attributes from Featureset.
	label = ForeignKeyField(Label, backref='features')

	def create_from_dataset_columns(
		dataset_id:int
		,column_names:list
		,label_id:int
	):
		d = Dataset.get_by_id(dataset_id)
		
		f_cols = column_names
		d_cols = d.column_names
		# Test that all Featureset columns exist in the Dataset and vice versa.
		all_f_cols_found = all(i in d_cols for i in f_cols)
		# The whole transaction is dirty if it isn't.
		if all_f_cols_found:
			tags=[]
			l = Label.get_by_id(label_id)
			l_col = l.column_name
			contains_label = l_col in f_cols
			if contains_label:
				raise ValueError("\nError - Label column `" + l_col + "` found within Featureset columns provided.\nYou cannot include the Label column in a Featureset of that Label.\n")
			else:
				# Prior to checking the reverse, remove the Label column.
				d_cols.remove(l_col)
				all_d_cols_found_but_label = all(i in f_cols for i in d_cols)
				d_cols.append(l_col)
				if all_d_cols_found_but_label:
					tags.append("all_dataset_features_except_label")
				else:
					tags.append("not_all_dataset_features")		

			f = Supervisedset.create(
				dataset=d
				,label=l
				,column_names=column_names
				,tags=tags
			)
			return f
		else:
			print("\nError - Could not find all of the provided column names in `Dataset.column_names`.\n" + " ".join(f_cols) + "\n")
			return None

	def create_all_columns_except_label(
		dataset_id:int
		,label_id:int
	):
		d = Dataset.get_by_id(dataset_id)
		l = Label.get_by_id(label_id)

		label_col = l.column_name
		dataset_cols = d.column_names
		# This is overwrites the list, excluding the label.
		dataset_cols.remove(label_col)

		s = Supervisedset.create_from_dataset_columns(
			dataset_id = dataset_id
			,column_names = dataset_cols
			,label_id = label_id
		)
		return s

class Unsupervisedset(Featureset):
	"""
	- Inherits attributes from Featureset. 
	- PCA components vary across featuresets. When different column combinations are used the same column will have different component values.
	"""

	def create_from_dataset_columns(
		dataset_id:int
		,column_names:list

	):
		d = Dataset.get_by_id(dataset_id)
		
		f_cols = column_names
		d_cols = d.column_names
		# Test that all Featureset columns exist in the Dataset and vice versa.
		all_f_cols_found = all(i in d_cols for i in f_cols)
		# The whole transaction is dirty if it isn't.
		if all_f_cols_found:
			tags=[]
			
			all_d_cols_found = all(i in f_cols for i in d_cols)
			if all_d_cols_found:
				tags.append("all_dataset_columns")
			else:
				tags.append("not_all_dataset_columns")		

			u = Unsupervisedset.create(
				dataset=d
				,column_names=column_names
				,tags=tags
			)
			return u
		else:
			print("\nError - Could not find all of the provided column names in `Dataset.column_names`.\n" + " ".join(f_cols) + "\n")
			return None

	def create_all_columns(
		dataset_id:int
	):
		d = Dataset.get_by_id(dataset_id)

		dataset_cols = d.column_names

		u = Unsupervisedset.create_from_dataset_columns(
			dataset_id = dataset_id
			,column_names = dataset_cols
		)
		return u
