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
		db.create_tables([Job, Dataset, Label, Featureset, Splitset, Algorithm])
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




class Dataset(BaseModel):
	name = CharField()
	data = BlobField()
	category = CharField() #tabular, image, longitudinal
	file_format = CharField()
	is_compressed = BooleanField()
	columns= JSONField()
	#is_shuffled= BooleanField()#False

	def from_file(
		path:str
		,file_format:str
		,name:str = None
		,perform_gzip:bool = True
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


	def to_pandas(
		id:int
		,columns:list = None
		,samples:list = None
	):
		"""
		- After unzipping `gzip.open()`, bytesio still needed to be read into PyArrow before being read into Pandas.
		- All methods return all columns by default if they receive None: 
		  `pc.read_csv(read_options.column_names)`, `pa.read_table()`, `pd.read_csv(uscols)`, `pd.read_parquet(columns)`
		"""
		d = Dataset.get_by_id(id)
		is_compressed = d.is_compressed
		ff = d.file_format
		
		# When user provides only 1 column and forgets to [] it (e.g. the label column).
		if type(columns) == str:
				columns = [columns]

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
						,sep = '\t'
						,usecols = columns)
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
					,columns = columns)
		
		if samples is not None:
			df = df.iloc[samples]

		return df


	def to_numpy(
		id:int
		,columns:list = None
		,samples:list = None
	):
		df = Dataset.to_pandas(id=id, columns=columns, samples=samples)
		arr = df.to_numpy()
		return arr

	"""
	Future:
	- Read as Tensors (pytorch and tf)? Or will numpy suffice?
	- Longitudinal data?
	- Images?
	"""
	"""
	def from_pandas():
		read as arrow
		save as tsv

	def from_numpy():
		read as arrow
		save as tsv
	"""

	def make_label(id:int, column:str):
		l = Label.from_dataset(dataset_id=id, column=column)
		return l


	def fetch_label_by_name(id:int, label_name:str):
		d = Dataset.get_by_id(id)
		try:
			matching_label = d.labels.select().where(Label.column==label_name)[0]
		except:
			raise ValueError("\nYikes - there is no Label with a `column` attribute named '" + label_name + "'\n")
		return matching_label


	def make_featureset(
		id:int
		, include_columns:list = None
		, exclude_columns:list = None
	):

		f = Featureset.from_dataset(
			dataset_id = id
			, include_columns = include_columns
			, exclude_columns = exclude_columns
			#Future: runPCA, correlationmatrix, feature_importance
		)
		return f




class Label(BaseModel):
	column=CharField()
	
	dataset = ForeignKeyField(Dataset, backref='labels')
	
	def from_dataset(dataset_id:int, column:str):
		d = Dataset.get_by_id(dataset_id)

		# check for duplicates
		d_labels = d.labels
		d_labels_col = [l.column for l in d_labels]
		if column in d_labels_col:
			raise ValueError("\nYikes - This Dataset already has a Label with target column named '" + column + "'.\nCannot create duplicate Label.\n")

		# verify that the column exists
		d_columns = d.columns
		column_found = column in d_columns
		if column_found:
			l = Label.create(dataset=d, column=column)
			return l
		else:
			raise ValueError("\nYikes - Column name '" + column + "' not found in `Dataset.columns`.\n")


	def to_pandas(id:int, samples:list=None):
		l = Label.get_by_id(id)
		l_col = l.column
		dataset_id = l.dataset.id
		
		lf = Dataset.to_pandas(
			id = dataset_id
			,columns = l_col
			,samples = samples
		)
		return lf


	def to_numpy(id:int, samples:list=None):
		lf = Label.to_pandas(id=id, samples=samples)
		l_arr = lf.to_numpy()
		return l_arr




class Featureset(BaseModel):
	"""
	- Remember, a Featureset is just a record of the columns being used.
	- Decided not to go w subclasses of Unsupervised and Supervised because that would complicate the SDK for the user,
	  and it essentially forked every downstream model into two subclasses.
	- So the ForeignKey on label is optional:
	  http://docs.peewee-orm.com/en/latest/peewee/api.html?highlight=deferredforeign#DeferredForeignKey
	- PCA components vary across featuresets. When different columns are used those columns have different component values.
	"""
	columns = JSONField()
	columns_excluded = JSONField(null=True)
	dataset = ForeignKeyField(Dataset, backref='featuresets')


	def from_dataset(
		dataset_id:int
		,include_columns:list=None
		,exclude_columns:list=None
		#Future: runPCA #,run_pca:boolean=False # triggers PCA analysis of all columns
	):

		d = Dataset.get_by_id(dataset_id)
		d_cols = d.columns

		if (include_columns is not None) and (exclude_columns is not None):
			raise ValueError("\nYikes - You can set either `include_columns` or `exclude_columns`, but not both.\n")

		if (include_columns is not None):
			# check columns exist
			all_cols_found = all(col in d_cols for col in include_columns)
			if not all_cols_found:
				raise ValueError("\nYikes - You specified `include_columns` that do not exist in the Dataset.\n")
			# inclusion
			columns = include_columns
			# exclusion
			columns_excluded = d_cols
			for col in include_columns:
				columns_excluded.remove(col)

		elif (exclude_columns is not None):
			all_cols_found = all(col in d_cols for col in exclude_columns)
			if not all_cols_found:
				raise ValueError("\nYikes - You specified `exclude_columns` that do not exist in the Dataset.\n")
			# exclusion
			columns_excluded = exclude_columns
			# inclusion
			columns = d_cols
			for col in exclude_columns:
				columns.remove(col)
			if not columns:
				raise ValueError("\nYikes - You cannot exclude every column in the Dataset.\n")
		else:
			columns = d_cols
			columns_excluded = None

		"""
		Check that this Dataset does not already have a Featureset that is exactly the same.
		Less entries in `excluded_columns` so maybe it's faster to compare.
		"""
		if columns_excluded is not None:
			cols_aplha = sorted(columns_excluded)
		else:
			cols_aplha = None
		d_featuresets = d.featuresets
		count = d_featuresets.count()
		if count > 0:
			for f in d_featuresets:
				f_id = str(f.id)
				f_cols = f.columns_excluded
				if f_cols is not None:
					f_cols_alpha = sorted(f_cols)
				else:
					f_cols_alpha = None
				if cols_aplha == f_cols_alpha:
					raise ValueError("\nYikes - This Dataset already has Featureset <id:" + f_id + "> with the same columns.\nCannot create duplicate.\n")

		f = Featureset.create(
			dataset = d
			, columns = columns
			, columns_excluded = columns_excluded
		)
		return f


	def to_pandas(id:int, samples:list=None):
		f = Featureset.get_by_id(id)
		f_cols = f.columns
		dataset_id = f.dataset.id
		
		ff = Dataset.to_pandas(
			id = dataset_id
			,columns = f_cols
			,samples = samples
		)
		return ff


	def to_numpy(id:int, samples:list=None):
		ff = Featureset.to_pandas(id=id, samples=samples)
		f_arr = ff.to_numpy()
		return f_arr


	def make_splitset(
		id:int
		,size_test:float = None
		,size_validation:float = None
		,fold_count:int = 1 #Future: folds
	):
		s = Splitset.from_featureset(
			featureset_id = id
			,size_test = size_test
			,size_validation = size_validation
			,fold_count = 1 #Future: folds
		)
		return s




class Splitset(BaseModel):
	"""
	- Belongs to a Featureset, not a Dataset, because the samples selected vary based on the stratification of the features during the split,
	  and a Featureset already has a Dataset anyways.
	  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
	- Can this also be supervised and unsupervised?
	"""
	samples = JSONField()
	sizes = JSONField()
	is_validated = BooleanField()
	is_folded = BooleanField()
	fold_count = IntegerField() #Future: folds # validate too many folds? try setting between 3-10 depending on the size of your data.

	featureset = ForeignKeyField(Featureset, backref='splitsets')
	label = ForeignKeyField(Label, deferrable='INITIALLY DEFERRED', null=True, backref='splitsets')
	

	def from_featureset(
		featureset_id:int
		,size_test:float = None
		,size_validation:float = None
		,fold_count:int = 1
	):

		if size_test is not None:
			if (size_test <= 0.0) or (size_test >= 1.0):
				raise ValueError("\nYikes - `size_test` must be between 0.0 and 1.0\n")
		
		if (size_validation is not None) and (size_test is None):
			raise ValueError("\nYikes - you specified a `size_validation` without setting a `size_test`.")

		if size_validation is not None:
			if (size_validation <= 0.0) or (size_validation >= 1.0):
				raise ValueError("\nYikes - `size_test` must be between 0.0 and 1.0\n")
			sum_test_val = size_validation + size_test
			if sum_test_val >= 1.0:
				raise ValueError("\nYikes - Sum of `size_test` + `size_test` must be between 0.0 and 1.0 to leave room for training set.\n")
			"""
			Have to run train_test_split twice do the math to figure out the size of 2nd split.
			Let's say I want {train:0.67, validation:0.13, test:0.20}
			The first test_size is 20% which leaves 80% of the original data to be split into validation and training data.
			(1.0/(1.0-0.20))*0.13 = 0.1625
			"""
			pct_for_2nd_split = (1.0/(1.0-size_test))*size_validation
			is_validated = True
		else:
			is_validated = False

		f = Featureset.get_by_id(featureset_id)
		f_cols = f.columns

		# Feature data to be split.
		d_id = f.dataset.id
		arr_f = Dataset.to_numpy(id=d_id, columns=f_cols)

		"""
		Simulate an index to be split alongside features and labels
		in order to keep track of the samples being used in the resulting splits.
		"""
		row_count = arr_f.shape[0]
		arr_idx = np.arange(row_count)
		
		samples = {}
		sizes = {}

		l = f.label
		if l is None:
			# Unsupervised
			if (size_test is not None) or (size_validation is not None):
				raise ValueError("\nYikes - Unsupervised Featuresets support neither test nor validation splits.\nSet both `size_test` and `size_validation` as `None` for this Featureset.\n")
			else:
				indices_lst_train = arr_idx.tolist()
				samples["train"] = indices_lst_train
				sizes["train"] = {"percent": 1.00, "count": row_count}
		else:
			# Supervised
			if size_test is None:
				size_test = 0.25

			l_id = l.id
			l_col = l.column 
			arr_l = Dataset.to_numpy(id=d_id, columns=[l_col])

			# `sklearn.model_selection.train_test_split` = https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
			features_train, features_test, labels_train, labels_test, indices_train, indices_test = train_test_split(
				arr_f, arr_l, arr_idx
				,test_size = size_test
				,stratify = arr_l
			)

			if size_validation is not None:
				features_train, features_validation, labels_train, labels_validation, indices_train, indices_validation = train_test_split(
					features_train, labels_train, indices_train
					,test_size = pct_for_2nd_split
					,stratify = labels_train
				)
				indices_lst_validation = indices_validation.tolist()
				samples["validation"] = indices_lst_validation

			indices_lst_train, indices_lst_test  = indices_train.tolist(), indices_test.tolist()
			samples["train"] = indices_lst_train
			samples["test"] = indices_lst_test

			size_train = 1.0 - size_test
			if size_validation is not None:
				size_train -= size_validation
				count_validation = len(indices_lst_validation)
				sizes["validation"] =  {"percent": size_validation, "count": count_validation}
			
			count_test = len(indices_lst_test)
			count_train = len(indices_lst_train)
			sizes["test"] = {"percent": size_test, "count": count_test}
			sizes["train"] = {"percent": size_train, "count": count_train}

		if fold_count < 2:
			is_folded = False

		s = Splitset.create(
			featureset = f
			,label = l
			,samples = samples
			,sizes = sizes
			,is_validated = is_validated
			,is_folded = is_folded
			,fold_count = fold_count
		)
		return s


	def to_pandas(id:int, splits:list=None):
		s = Splitset.get_by_id(id)

		if splits is not None:
			if len(splits) == 0:
				raise ValueError("Yikes - `splits:list` is an empty list.\nIt can be None, which defaults to all splits, but it can't not empty.")
		else:
			splits = list(s.samples.keys())

		f = s.featureset
		f_supervision = f.supervision

		split_frames = {}
		
		if f_supervision == "unsupervised":
			set_dct = {"features": None}
		else:
			set_dct = {"features": None, "labels": None}

		# Flag:Optimize (switch to generators for memory usage)
		# split_names = train, test, validation
		for split_name in splits:
			split_frames[split_name] = set_dct

			samples = s.samples[split_name]

			ff = f.to_pandas(samples=samples)
			split_frames[split_name]["features"] = ff

			if f_supervision == "supervised":
				l = f.label
				lf = l.to_pandas(samples=samples)
				split_frames[split_name]["labels"] = lf
		return split_frames


	def to_numpy(id:int, splits:list=None):
		"""
		Flag:Optimize 
		- Worried it's holding all dataframes and arrays in memory.
		- Generators to access one [key][set] at a time?
		"""
		split_frames = Splitset.to_pandas(id=id, splits=splits)

		# packing a fresh dct because I don't trust updating dcts.
		split_arrs = {}

		split_keys = split_frames.keys()
		# split_names = train, test, validation
		for split in split_keys:
			set_keys = split_frames[split].keys()
			split_arrs[split] = {}

			# set_names = features, labels
			for set_name in set_keys:
				frame = split_frames[split][set_name]
				arr = frame.to_numpy()
				split_arrs[split][set_name] = arr
				del frame # prevent memory usage doubling.

		return split_arrs




class Algorithm(BaseModel):
	name = CharField()




class Job(BaseModel):
	status = CharField()
