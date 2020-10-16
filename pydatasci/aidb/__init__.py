name = "aidb"

import os, sqlite3, io, gzip, zlib, random
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
from sklearn.model_selection import train_test_split, StratifiedKFold

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
		db.create_tables([Job, Dataset, Label, Featureset, Splitset, Foldset, Algorithm])
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
	shape = JSONField()
	dtype = JSONField(null=True)
	file_format = CharField()
	is_compressed = BooleanField()
	columns = JSONField()
	#is_shuffled= BooleanField()#False

	def from_file(
		path:str
		, file_format:str = None
		, name:str = None
		, perform_gzip:bool = True
		, dtype:dict = None
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
		Dataset.check_file_format(file_format)

		if name is None:
			name=path
		if perform_gzip is None:
			perform_gzip=True

		# File formats.
		if (file_format == 'tsv') or (file_format is None):
			parse_opt = pc.ParseOptions(delimiter='\t')
			tbl = pc.read_csv(path, parse_options=parse_opt)
			file_format = 'tsv'
		elif (file_format == 'csv'):
			parse_opt = pc.ParseOptions(delimiter=',')
			tbl = pc.read_csv(path)
		elif (file_format == 'parquet'):
			tbl = pq.read_table(path)

		#ToDo - handle columns with no name.
		columns = tbl.column_names
		shape = {}
		shape['rows'], shape['columns'],  = tbl.num_rows, tbl.num_columns

		with open(path, "rb") as f:
			bytesio = io.BytesIO(f.read())
			data = bytesio.getvalue()
			data, is_compressed = Dataset.compress_or_not(data, perform_gzip)

		d = Dataset.create(
			name = name
			, data = data
			, shape = shape
			, dtype = dtype
			, file_format = file_format
			, is_compressed = is_compressed
			, columns = columns
		)
		return d


	def from_pandas(
		dataframe
		, name:str = None
		, file_format:str = None
		, perform_gzip:bool = True
		, dtype:dict = None
		, rename_columns:list = None
	):
		if dataframe.empty:
			raise ValueError("\nYikes - The dataframe you provided is empty according to `df.empty`")

		Dataset.check_file_format(file_format)
		Dataset.check_column_count(user_columns=rename_columns, structure=dataframe)

		shape = {}
		shape['rows'], shape['columns'] = dataframe.shape[0], dataframe.shape[1]

		if dtype is None:
			dct_types = dataframe.dtypes.to_dict()
			# convert the `dtype('float64')` to strings
			keys_values = dct_types.items()
			dtype = {k: str(v) for k, v in keys_values}
		
		# Passes in user-defined columns in case they are specified
		dataframe, columns = Dataset.pandas_stringify_columns(df=dataframe, columns=rename_columns)

		# https://stackoverflow.com/a/25632711
		buff = io.StringIO()
		if (file_format == 'tsv') or (file_format is None):
			dataframe.to_csv(buff, index=False, sep='\t')
			buff_string = buff.getvalue()
			data = bytes(buff_string, 'utf-8')
			file_format = 'tsv'
		elif (file_format == 'csv'):
			dataframe.to_csv(buff, index=False, sep=',')
			buff_string = buff.getvalue()
			data = bytes(buff_string, 'utf-8')
		elif (file_format == 'parquet'):
			buff = io.BytesIO()
			dataframe.to_parquet(buff) 
			data = buff.getvalue()

		data, is_compressed = Dataset.compress_or_not(data, perform_gzip)

		if name is None:
			name = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p") + "." + file_format

		d = Dataset.create(
			name = name
			, data = data
			, shape = shape
			, dtype = dtype
			, file_format = file_format
			, is_compressed = is_compressed
			, columns = columns
		)
		return d


	def from_numpy(
		ndarray
		, name:str = None
		, file_format:str = None
		, perform_gzip:bool = True
		, column_names:list = None #pd.Dataframe param
		, dtype:str = None #pd.Dataframe param
	):
		Dataset.check_file_format(file_format)
		Dataset.check_column_count(user_columns=column_names, structure=ndarray)

		if ndarray.size == 0:
			raise ValueError("\nYikes - The ndarray you provided is empty: `ndarray.size == 0`.\n")

		# check if it is an ndarray as opposed to structuredArray
		if (ndarray.dtype.names is None):
			if False in np.isnan(ndarray[0]):
			    pass
			else:
				ndarray = np.delete(ndarray, 0, axis=0)
				print("\nInfo - The entire first row of your array is 'nan', so we deleted this row during ingestion.\n")
			
			col_names = ndarray.dtype.names
			if (col_names is None) and (column_names is None):
				# generate string-based column names to feed to pandas
				col_count = ndarray.shape[1]
				column_names = [str(i) for i in range(col_count)]
				print("\nInfo - You didn't provide any column names for your array, so we generated them for you.\ncolumn_names: " + str(column_names) + "\n" )
			
		shape = {}
		shape['rows'], shape['columns'] = ndarray.shape[0], ndarray.shape[1]

		df = pd.DataFrame(
			data = ndarray
			, columns = column_names
			, dtype = dtype # pandas only accepts a single str. pandas infers if None.
		)
		del ndarray
		
		d = Dataset.from_pandas(
			dataframe = df
			, name = name
			, file_format = file_format
			, perform_gzip = perform_gzip
			, dtype = None # numpy dtype handled when making df above.
		)
		return d


	def to_pandas(
		id:int
		, columns:list = None
		, samples:list = None
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
						, sep = '\t'
						, usecols = columns)
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

		d_dtype = d.dtype
		if d_dtype is not None:
			if (type(d_dtype) == dict):
				if columns is None:
					columns = d.columns
				# need to prune out the excluded columns from the dtype dict
				d_dtype_cols = list(d_dtype.keys())
				for col in d_dtype_cols:
					if col not in columns:
						del d_dtype[col]
			df = df.astype(d_dtype)

		return df


	def to_numpy(
		id:int
		,columns:list = None
		,samples:list = None
	):
		d = Dataset.get_by_id(id)
		# dtype is applied within `to_pandas()` function.
		df = Dataset.to_pandas(id=id, columns=columns, samples=samples)
		arr = df.to_numpy()
		return arr

	"""
	Future:
	- Read to_tensor (pytorch and tf)? Or will numpy suffice?
	"""

	def make_label(id:int, column:str):
		l = Label.from_dataset(dataset_id=id, column=column)
		return l


	def fetch_label_by_name(id:int, label_name:str):
		d = Dataset.get_by_id(id)
		matching_labels = d.labels.select().where(Label.column==label_name)
		count_matches = matching_labels.count()
		
		if count_matches > 0:
			matching_label = matching_labels[0]
		else:
			matching_label = None
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


	def pandas_stringify_columns(df, columns):
		cols_raw = df.columns.to_list()
		if columns is None:
			# in case the columns were a range of ints.
			cols_str = [str(c) for c in cols_raw]
		else:
			cols_str = columns
		# dict from 2 lists
		cols_dct = dict(zip(cols_raw, cols_str))
		
		df = df.rename(columns=cols_dct)
		columns = df.columns.to_list()
		return df, columns


	def check_file_format(file_format):
		accepted_formats = ['csv', 'tsv', 'parquet', None]
		if file_format not in accepted_formats:
			raise ValueError("\nYikes - Available file formats include uncompressed csv, tsv, and parquet.\nYour file format: " + file_format + "\n")


	def check_column_count(user_columns, structure):
		col_count = len(user_columns)
		structure_col_count = structure.shape[1]
		if col_count != structure_col_count:
			raise ValueError("\nYikes - The dataframe you provided has " + structure_col_count + "columns, but you provided " + col_count + "columns.\n")


	def compress_or_not(data, perform_gzip):
		if perform_gzip:
			data = gzip.compress(data)
			is_compressed=True
		else:
			is_compressed=False
		return data, perform_gzip





class Label(BaseModel):
	column = CharField()
	#probabilities = JSONField() #result of semi-supervised learning.
	
	dataset = ForeignKeyField(Dataset, backref='labels')
	
	def from_dataset(dataset_id:int, column:str):
		d = Dataset.get_by_id(dataset_id)

		# check for duplicates
		matching_label = d.fetch_label_by_name(label_name=column)
		if matching_label is not None:
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
			, columns = [l_col]
			, samples = samples
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
		, label_name:str = None
		, size_test:float = None
		, size_validation:float = None
	):
		s = Splitset.from_featureset(
			featureset_id = id
			, label_name = label_name
			, size_test = size_test
			, size_validation = size_validation
		)
		return s




class Splitset(BaseModel):
	"""
	- Belongs to a Featureset, not a Dataset, because the samples selected vary based on the stratification of the features during the split,
	  and a Featureset already has a Dataset anyways.
	"""
	samples = JSONField()
	sizes = JSONField()
	supervision = CharField()
	has_test = BooleanField()
	has_validation = BooleanField()

	featureset = ForeignKeyField(Featureset, backref='splitsets')
	label = ForeignKeyField(Label, deferrable='INITIALLY DEFERRED', null=True, backref='splitsets')
	

	def from_featureset(
		featureset_id:int
		, label_name:str = None
		, size_test:float = None
		, size_validation:float = None
		, continuous_bin_count:float = None
	):

		if size_test is not None:
			if (size_test <= 0.0) or (size_test >= 1.0):
				raise ValueError("\nYikes - `size_test` must be between 0.0 and 1.0\n")
			# Don't handle `has_test` here. Need to check label first.
			
		
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
			has_validation = True
		else:
			has_validation = False

		f = Featureset.get_by_id(featureset_id)
		f_cols = f.columns

		# Feature data to be split.
		d = f.dataset
		d_id = d.id
		arr_f = Dataset.to_numpy(id=d_id, columns=f_cols)

		"""
		Simulate an index to be split alongside features and labels
		in order to keep track of the samples being used in the resulting splits.
		"""
		row_count = arr_f.shape[0]
		arr_idx = np.arange(row_count)
		
		samples = {}
		sizes = {}

		if label_name is None:
			has_test = False
			supervision = "unsupervised"
			l = None
			if (size_test is not None) or (size_validation is not None):
				raise ValueError("\nYikes - Unsupervised Featuresets support neither test nor validation splits.\nSet both `size_test` and `size_validation` as `None` for this Featureset.\n")
			else:
				indices_lst_train = arr_idx.tolist()
				samples["train"] = indices_lst_train
				sizes["train"] = {"percent": 1.00, "count": row_count}
		else:
			# Splits generate different samples each time, so we do not need to prevent duplicates on the same label_name.
			l = Dataset.fetch_label_by_name(id=d_id, label_name=label_name)
			if l is None:
				raise ValueError("\nYikes - there is no Label with a `column` attribute named '" + label_name + "'\n")

			if size_test is None:
				size_test = 0.25
			has_test = True
			supervision = "supervised"

			l_id = l.id
			l_col = l.column 
			arr_l = Dataset.to_numpy(id=d_id, columns=[l_col])
			arr_l_dtype = arr_l.dtype

			if (arr_l_dtype == 'float32') or (arr_l_dtype == 'float64'):
				stratify1 = Splitset.continuous_bins(arr_l, continuous_bin_count)
			else:
				stratify1 = arr_l
			"""
			- `sklearn.model_selection.train_test_split` = https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
			- `shuffle` happens before the split. Although preserves a df's original index, we don't need to worry about that because we are providing our own indices.
			"""
			features_train, features_test, labels_train, labels_test, indices_train, indices_test = train_test_split(
				arr_f, arr_l, arr_idx
				, test_size = size_test
				, stratify = stratify1
				, shuffle = True
			)

			if size_validation is not None:
				if (arr_l_dtype == 'float32') or (arr_l_dtype == 'float64'):
					stratify2 = Splitset.continuous_bins(labels_train, continuous_bin_count)
				else:
					stratify2 = labels_train

				features_train, features_validation, labels_train, labels_validation, indices_train, indices_validation = train_test_split(
					features_train, labels_train, indices_train
					, test_size = pct_for_2nd_split
					, stratify = stratify2
					, shuffle = True
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

		s = Splitset.create(
			featureset = f
			, label = l
			, samples = samples
			, sizes = sizes
			, supervision = supervision
			, has_test = has_test
			, has_validation = has_validation
		)
		return s


	def to_pandas(id:int, splits:list=None):
		s = Splitset.get_by_id(id)

		if splits is not None:
			if len(splits) == 0:
				raise ValueError("\nYikes - `splits:list` is an empty list.\nIt can be None, which defaults to all splits, but it can't not empty.\n")
		else:
			splits = list(s.samples.keys())

		supervision = s.supervision
		f = s.featureset

		split_frames = {}

		# Flag:Optimize (switch to generators for memory usage)
		# split_names = train, test, validation
		for split_name in splits:
			
			# placeholder for the frames/arrays
			split_frames[split_name] = {}
			
			# fetch the sample indices for the split
			split_samples = s.samples[split_name]
			ff = f.to_pandas(samples=split_samples)
			split_frames[split_name]["features"] = ff

			if supervision == "supervised":
				l = s.label
				lf = l.to_pandas(samples=split_samples)
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


	def continuous_bins(array_to_bin, continuous_bin_count:int):
		if continuous_bin_count is None:
			continuous_bin_count = 4

		max = np.amax(array_to_bin)
		min = np.amin(array_to_bin)
		bins = np.linspace(start=min, stop=max, num=continuous_bin_count)
		flts_binned = np.digitize(array_to_bin, bins, right=True)
		return flts_binned

	def make_foldset(id:int, fold_count:int=None):
		foldset = Foldset.from_splitset(splitset_id=id, fold_count=fold_count)
		return foldset




class Foldset(BaseModel):
	folds = JSONField()
	fold_count = IntegerField()
	random_state = IntegerField()

	splitset = ForeignKeyField(Splitset, backref='foldsets')

	def from_splitset(
		splitset_id:int
		, fold_count:int = None
	):
		s = Splitset.get_by_id(splitset_id)
		new_random = False
		while new_random == False:
			random_state = random.randint(0, 4294967295) #2**32 - 1 inclusive
			matching_randoms = s.foldsets.select().where(Foldset.random_state==random_state)
			count_matches = matching_randoms.count()
			if count_matches == 0:
				new_random = True
		if fold_count is None:
			#ToDo - check the size of test. want like 30 in each fold
			fold_count = 5
		else:
			if fold_count < 2:
				raise ValueError("\nYikes - Cross validation requires multiple folds and you set `fold_count` < 2.\n")

		# get the training indices. the values of the features don't matter, only labels needed for stratification.
		arr_train_indices = s.samples["train"]
		arr_train_labels = s.label.to_numpy(samples=arr_train_indices)

		train_count = len(arr_train_indices)
		if train_count % fold_count != 0:
			print("\nAdvice - The length <" + train_count + "> of your training Split is not evenly divisible by the number of folds <" + fold_count + "> you specified.\nThere's a chance that this could lead to misleadingly low accuracy for the last Fold.")

		# This 'test' split that is untouched by the cross-validation process.
		index_holdout_test = s.samples["test"]
		if s.has_validation:
			index_holdout_validation = s.samples["validation"]

		skf = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=random_state)
		splitz_gen = skf.split(arr_train_indices, arr_train_labels)
		
		folds, i = {}, -1
		for index_train, index_validation in splitz_gen:
			i+=1
			if s.has_validation:
				folds[i] = {
					"folds_train_combined": index_train.tolist()
					, "fold_validation": index_validation.tolist()
					, "holdout_validation": index_holdout_validation
					, "holdout_test": index_holdout_test
				}
			else:
				folds[i] = {
					"folds_train_combined": index_train.tolist()
					, "fold_validation": index_validation.tolist()
					, "holdout_test": index_holdout_test
				}

		foldset = Foldset.create(
			folds = folds
			, fold_count = fold_count
			, random_state = random_state
		)
		return foldset



	def to_pandas(id:int, fold_index:int):
		"""
		- Only going to support fetching by a single fold_index because even that 
		  will have 3 dataframes/ arrays. Can loop easily on `range(fold_count)`.
		"""
		foldset = Foldset.get_by_id(id=id)
		fold_count = foldset.fold_count

		if fold_index is not None:
			if (0 > fold_index) or (fold_index > fold_count):
				raise ValueError("\nYikes - This Foldset <id:" + str(id) +  "> has indices between 0 and " + str(fold_count) + "\n")

		"""
		s = foldset.splitset
		supervision = s.supervision
		f = s.featureset

		fold_frames = {}

		# Flag:Optimize (switch to generators for memory usage)
		# fold_frames = train_folds_combined, validation_fold, test_holdout
		for split_name in splits:
			
			# placeholder for the frames/arrays
			split_frames[split_name] = {}
			
			# fetch the sample indices for the split
			split_samples = s.samples[split_name]
			ff = f.to_pandas(samples=split_samples)
			split_frames[split_name]["features"] = ff

			if supervision == "supervised":
				l = s.label
				lf = l.to_pandas(samples=split_samples)
				split_frames[split_name]["labels"] = lf
		"""


		fold_frames = Dataset.
		return fold_frames

	# def to_numpy():
	"""


class Algorithm(BaseModel):
	name = CharField()




class Job(BaseModel):
	status = CharField()
