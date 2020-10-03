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
from sklearn.model_selection import train_test_split

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
		db.create_tables([Job, Dataset, Label, Featureset, Foldset])
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
	columns= JSONField()
	#is_shuffled= BooleanField()#False

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

			#ToDo prevent ff combos like '.csv' with 'parquet' vice versa.

			# File formats.
			if file_format == 'csv':
				parse_opt = pc.ParseOptions(delimiter=',')
				tbl = pc.read_csv(path)
			elif file_format == 'tsv':
				parse_opt = pc.ParseOptions(delimiter='\t')
				tbl = pc.read_csv(path, parse_options=parse_opt)
			elif file_format == 'parquet':
				tbl = pq.read_table(path)

			# Future: handle other formats like image.
			category = 'tabular'

			#ToDo - handle columns with no name.
			columns = tbl.column_names

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
				,columns = columns
				,category = category
			)
			return d


	def read_to_pandas(
		id:int
		,columns:list=None
	):
		"""
		- After unzipping `gzip.open()`, bytesio still needed to be read into PyArrow before being read into Pandas.
		- All methods return all columns by default if they receive None: 
		  `pc.read_csv(read_options.column_names)`, `pa.read_table()`, `pd.read_csv(uscols)`, `pd.read_parquet(columns)`
		"""
		d = Dataset.get_by_id(id)
		is_compressed = d.is_compressed
		ff = d.file_format
		
		data = d.data
		bytesio_data = io.BytesIO(data)
		if (ff == 'csv') or (ff == 'tsv'):
			# `pc.ReadOptions.column_names` verifies the existence of the names, does not filter for them.
			if is_compressed:
				bytesio_csv = gzip.open(bytesio_data)
				if ff == 'tsv':
					parse_opt = pc.ParseOptions(delimiter='\t')
					tbl = pc.read_csv(bytesio_csv, parse_options=parse_opt)
				else:
					tbl = pc.read_csv(bytesio_csv)
				df = tbl.to_pandas()
				if columns is not None:
					df = df.filter(columns)
			else:
				if ff == 'tsv':
					df = pd.read_csv(
						bytesio_data
						,sep='\t'
						,usecols=columns)
				else:
					df = pd.read_csv(bytesio_data, usecols=columns)
		elif ff == 'parquet':
			if is_compressed:
				bytesio_parquet = gzip.open(bytesio_data)
				tbl = pq.read_table(bytesio_parquet, columns=columns)
				df = tbl.to_pandas()
			else:
				df = pd.read_parquet(
					bytesio_data
					,columns=columns)
		return df


	def read_to_numpy(
		id:int
		,columns:list=None
	):
		"""
		- Returns a NumPy structured array: https://numpy.org/doc/stable/user/basics.rec.html
		- Started implementing `np.genfromtxt(bytesio_data, names=True, delimiter=',')`, but just switched to Pandas.
		- There doesn't seem to be a direct Parquet to NumPy, so have to convert through PyArrow or Pandas.
		"""
		df = Dataset.read_to_pandas(id, columns=columns)
		#arr = df.to_records(index=False)
		arr = df.to_numpy()
		return arr

	"""
	Future:
	- Read as Tensors (pytorch and tf)? Or will numpy suffice?
	- Longitudinal data?
	- Images?
	"""

	"""
	def create_dataset_from_pandas():
		read as arrow
		save as parquet from some kind of buffer?

	def create_dataset_from_numpy():
		read as arrow
		save as parquet from some kind of buffer?
	"""

class Label(BaseModel):
	column=CharField()
	
	dataset = ForeignKeyField(Dataset, backref='labels')
	
	def create_from_dataset(
		dataset_id:int
		,column:str
	):
		d = Dataset.get_by_id(dataset_id)

		# verify that the column exists
		d_columns = d.columns
		column_found = column in d_columns
		if column_found:
			l = Label.create(
				dataset=d
				,column=column
			)
			return l
		else:
			print("Error - Column name not found in `Dataset.columns`.")
			return None


class Featureset(BaseModel):
	"""
	- Remember, a Featureset is just a record of the columns being used.
	- Decided not to go w subclasses of Unsupervised and Supervised because that would complicate the SDK for the user,
	  and it essentially made every downstream model forked into subclasses.
	- So the ForeignKey on label is optional:
	  http://docs.peewee-orm.com/en/latest/peewee/api.html?highlight=deferredforeign#DeferredForeignKey
	- PCA components vary across featuresets. When different columns are used those columns have different component values.
	"""
	supervision = CharField() #supervised, unsupervised
	columns = JSONField()
	contains_all_columns = BooleanField()
	dataset = ForeignKeyField(Dataset, backref='featuresets')
	label = ForeignKeyField(Label, deferrable='INITIALLY DEFERRED', null=True, backref='featuresets')


	def create_from_dataset_columns(
		dataset_id:int
		,columns:list
		,label_id:int=None # triggers `supervision = unsupervised`
	):
		d = Dataset.get_by_id(dataset_id)

		f_cols = columns
		d_cols = d.columns
		# Test that all Featureset columns exist in the Dataset, but not yet vice versa.
		# The whole transaction is invalid if this is False.
		all_f_cols_found = all(col in d_cols for col in f_cols)
		if all_f_cols_found:
			if label_id is not None:
				l = Label.get_by_id(label_id)
				l_col = l.column
				contains_label = l_col in f_cols
				if contains_label:
					print("\nError - Label column `" + l_col + "` found within Featureset columns provided.\nYou cannot include the Label column in a Featureset of that Label.\n")
				else:
					supervision = "supervised"
					# Prior to checking the reverse, remove the Label column.
					d_cols.remove(l_col)
					all_d_cols_found_but_label = all(i in f_cols for i in d_cols)
					d_cols.append(l_col)
					if all_d_cols_found_but_label:
						contains_all_columns = True
					else:
						contains_all_columns = False
			else:
				l = None
				supervision = "unsupervised"
				all_d_cols_found = all(i in f_cols for i in d_cols)
				if all_d_cols_found:
					contains_all_columns = True
				else:
					contains_all_columns = False			

			f = Featureset.create(
				dataset=d
				,label=l
				,columns=columns
				,supervision=supervision
				,contains_all_columns=contains_all_columns
			)
			return f
		else:
			print("\nError - Could not find all of the provided column names in `Dataset.columns`.\n" + " ".join(d_cols) + "\n")


	def create_all_columns(
		dataset_id:int
		,label_id:int=None
	):
		d = Dataset.get_by_id(dataset_id)
		dataset_cols = d.columns

		if label_id is not None:
			l = Label.get_by_id(label_id)
			label_col = l.column
			# Overwrites the original list.
			dataset_cols.remove(label_col)

		f = Featureset.create_from_dataset_columns(
			dataset_id = dataset_id
			,columns = dataset_cols
			,label_id = label_id
		)
		return f


class Foldset(BaseModel):
	"""
	- Belongs to a Featureset, not a Dataset, because the rows selected vary based on the stratification of the features during the split,
	  and a Featureset already has a Dataset anyways.
	  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
	"""
	#is_validation_set_used=BooleanField()
	#is_multiple_folds=BooleanField()
	#fold_count=IntegerField() # validate too many folds? try setting between 3-10 depending on the size of your data.
	#folds_and_splits=JSONField()

	featureset = ForeignKeyField(Featureset, backref='foldset')
	label = ForeignKeyField(Label, deferrable='INITIALLY DEFERRED', null=True, backref='foldsets')


	"""
	# this is whatt the folds JSON looks like:
		0:{
			train:{ 
			row_indices: [],
			label_distribution: {#:%}
			size_samples_specified: 0.0
			size_samples_actual: 0.0
		validation:{,
		test: {
		}
	"""
	
	def create_from_featureset(
		featureset_id:int
		#size_train
		#size_test
		#size_validation
	):
		f = Featureset.get_by_id(featureset_id)
		f_cols = f.columns

		d_id = f.dataset.id
		df_f = Dataset.read_to_numpy(id=d_id, columns=f_cols)

		l = f.label
		if l is not None:
			l_col = l.column 
			df_l = Dataset.read_to_numpy(id=d_id, columns=l_col)
			# <------- it doesnt like when the y is a df
			# nested arrays don't have a shape -_-

		size_test=0.30

		features_train, features_test, labels_train, labels_test = train_test_split(
    		df_f, 
    		df_l,
    		test_size=0.30
    	)

		return features_train, features_test, labels_train, labels_test

	#def create_splits_from_featureset():


#class Job(BaseModel):
	#this will also have the deferrable key to label. 
