name = "aidb"

import os, sqlite3, io, gzip
from datetime import datetime

#orm
from peewee import *
from playhouse.dataset import DataSet
from playhouse.sqlite_ext import SqliteExtDatabase, JSONField
#data ingestion
import pyarrow
from pyarrow import parquet as pq
from pyarrow.csv import read_csv

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
	# ToDo - Could let the user specify their own db name, for import tutorials. Could check if passed as an argument to create_config?
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
	name:str=None,
	perform_gzip:bool=True
):
	"""
	Does not accept compressed files.
	"""
	
	#ToDo column names. pyarrow layer to handle files?
	#ToDo csv,tsv,parquet,gzip
	
	if name is None:
		name=path

	tbl = pyarrow.csv.read_csv(path)
	column_names = tbl.column_names

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
		is_compressed = is_compressed,
		column_names = column_names
	)

	return d


#def create_dataset_from_pandas():
#def create_dataset_from_numpy():


def read_dataset():
	pass
	# handle types
	# can pyarrow read csv in bytes?


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
	is_compressed = BooleanField()
	column_names= JSONField()
	#storage_format = CharField() #sqlite_blob, path_single, path_partitioned
	#original_format = pandas, numpy, file_parquet, file_parquet_gzip, file_parquet_partitions, file_csv, file_tsv
	#compression = CharField()
