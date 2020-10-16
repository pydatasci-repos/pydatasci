![PyDataSci (wide)](/images/logo_pds_wide.png)

---

# Value Proposition
*PyDataSci* is an open source, autoML tool that keeps track of the moving parts of machine learning so that data scientists can perform best practice ML without the coding overhead.

## TLDR
```
pip install pydatasci

import pydatasci as pds
pds.create_folder()
pds.create_config()

from pydatasci import aidb
aidb.create_db()
```

## Mission
* **Reproducibly Persistent & Embedded**<br />No more black boxes. No more screenshotting parameters and loss-accuracy graphs. A record of every: dataset, feature, sample (index), split, fold, parameter, run, model, and result - is automatically persisted in a lightweight, file-based database that is setup when you import the package. So hypertune to your heart's content, visually compare models, and know that you've found the best one with the proof to back it up.<br /><br />

* **Local-First**<br />We empower non-cloud users: the academic/ institute HPCers, the private clouders, the remote server SSH'ers, and everyday desktop hackers - with the same quality ML tooling as present in public clouds (e.g. AWS SageMaker).<br /><br />

* **Code-Integrated**<br />We don’t disrupt the natural workflow of users by forcing them into the confines of a GUI app or specific IDE. Instead, we weave automated tracking into their existing code so that *PyDataSci* is compatible with any data science toolset.<br /><br />

* **Scale-Em-If-You-Gottem**<br />Queue many hypertuning jobs locally, or run big jobs in parallel in the cloud by setting `cloud_queue = True`.<br /><br />


## Functionality
- [Done] Compress a dataset (csv, tsv, parquet, pandas dataframe, numpy ndarray) to be analyzed.
- [Done] Split samples by index while treating validation sets (3rd split) and cross-folds (k-fold) as first-level citizens.
- Derive informative featuresets from that dataset using supervised and unsupervised methods.
- Queue hypertuning jobs and batches based on hyperparameter combinations.
- Evaluate and save the performance metrics of each model.
- Visually compare model metrics in Jupyter Notebooks with Plotly Dash to find the best one.
- Behind the scenes, stream rows from your datasets and use generators to keep a low memory footprint.
- Scale out to run cloud jobs in parallel by toggling `cloud_queue = True`.

![Ecosystem Banner (wide)](/images/ecosystem_banner.png)

## Painpoint Solved
At the time, I was deep in an unstable, remote Linux workspace trying to finish a meta-analysis of methods for interpreting neural network activation values as an alternative approach to predictions based on the traditional feedforward weighted sum. I was running so many variations of models from different versions of graph neural network algorithms, CNNs, LSTMs... the analysis was really starting to pile up. First I started taking screenshots of my loss-accuracy graphs and that worked fine for a while. Then I started taking screenshots of my hyper-params; I couldn't be bothered to write down every combination of parameters I was running and the performance metrics they spit out every time. But, then again, I hadn't generated confusion matrices to compare and I should really record my feature importance ranking... and then the wheels really fell off when I started questioning if my `df` in-memory was really the `df` I thought it was last week. 

Slamming my head on the keyboard a few times, I thought to myself "I don't want to do all of that..." So I did what any good hacker would do and started a full blown project around it. I had done the hardest part in figuring out the science, but this permuted world was just a mess when it came time to systematically prove it. It wasn't conducive to the scientific method. I had also been keeping an eye on other tools in the space. They seemed lacking in that they were either: cloud-only, dependent on an external database, the integration processes were too complex for data scientists/ statisticians/ researchers, the tech wasn't distributed properly, or they were just too proprietary/ walled garden/ biased toward corporate ecosystems.


## Community
*Much to automate there is. Simple it must be.* ML is a broad space with a lot of challenges to solve. Let us know if you want to get involved. We plan to host monthly dev jam sessions and data science lightning talks.

* **Data types to support:** tabular, time series, image, graph, audio, video, gaming.
* **Analysis types to support for each data type:** classification, regression, dimensionality reduction, feature engineering, recurrent, generative, NLP, reinforcement.

---

# Installation
Requires Python 3+ (check your deep learning library's Python requirements). You will only need to perform these steps the first time you use the package. 

Enter the following commands one-by-one and follow any instructions returned by the command prompt to resolve errors should they arise.

_Starting from the command line:_
```bash
$ pip install --upgrade pydatasci
$ python
```
_Once inside the Python shell:_
```python
>>> import pydatasci as pds
>>> pds.create_folder()
>>> pds.create_config()
>>> from pydatasci import aidb
>>> aidb.create_db()
```

> PyDataSci makes use of the Python package, `appdirs`, for an operating system (OS) agnostic location to store configuration and database files. This not only keeps your `$HOME` directory clean, but also helps prevent careless users from deleting your database. 
>
> The installation process checks not only that the corresponding appdirs folder exists on your system but also that you have the permissions neceessary to read from and write to that location. If these conditions are not met, then you will be provided instructions during the installation about how to create the folder and/ or grant yourself the appropriate permissions. 
>
> We have attempted to support both Windows (`icacls` permissions and backslashes `C:\\`) as well as POSIX including Mac and Linux including containers & Google Colab (`chmod letters` permissions and slashes `/`). Note: due to variations in the ordering of appdirs author and app directories in different OS', we do not make use of the appdirs `appauthor` directory, only the `appname` directory.
>
> If you run into trouble with the installation process on your OS, please submit a GitHub issue so that we can attempt to resolve, document, and release a fix as quickly as possible.
>
> _Installation Location Based on OS_<br />`import appdirs; appdirs.user_data_dir('pydatasci');`:
> * Mac: <br />`/Users/Username/Library/Application Support/pydatasci`<br /><br />
> * Linux - Alpine and Ubuntu: <br />`/root/.local/share/pydatasci`<br /><br />
> * Windows: <br />`C:\Users\Username\AppData\Local\pydatasci`
>
> `create_db()` is equivalent to a *migration* in Django or Rails in that it creates the tables found in the Object Relational Model (ORM). We use the [`peewee`](http://docs.peewee-orm.com/en/latest/peewee/models.html) ORM as it is simpler than SQLAlchemy, has good documentation, and found the project to be actively maintained (saw same-day GitHub response to issues on a Saturday). With the addition of Dash-Plotly, this will make for a full-stack experience that also works directly in an IDE like Jupyter or VS Code.


### Deleting & Recreating the Database
When deleting the database, you need to either reload the `aidb` module or restart the Python shell before you can attempt to recreate the database.

```python
>>> from pydatasci import aidb
>>> aidb.delete_db(True)
>>> from importlib import reload
>>> reload(aidb)
>>> create_db()
```

---

# Usage

If you've already completed the *Installation* section above, let's get started.

### 1. Import the Library
```python
import pydatasci as pds
from pydatasci import aidb
```

### 2. Ingest a `Dataset` as a compressed file.

Supported tabular file formats include CSV, [TSV](https://stackoverflow.com/a/9652858/5739514), [Apache Parquet](https://parquet.apache.org/documentation/latest/).

```python
# From files
dataset = aidb.Dataset.from_file(
	path = 'iris.tsv' # files must have column names as their first row
	, file_format = 'tsv'
	, name = 'tab-separated plants'
	, perform_gzip = True
	, dtype = 'float64' # or a dict or dtype by column name.
)

# From in-memory structures
dataset = aidb.Dataset.from_pandas(
	dataframe = df
	, file_format = 'csv'
	, name = 'comma-separated plants'
	, perform_gzip = False
	, dtype = None # None infers from dataframe provided
	, rename_columns = None
)

dataset = aidb.Dataset.from_numpy(
    ndarray = arr
	, file_format = 'parquet'
	, name = 'chunking plants'
	, perform_gzip = True
	, dtype = None # feeds pd.Dataframe(dtype)
	, column_names = None # feeds pd.Dataframe(columns)
)
```
> Apart from `read_numpy()`, it's best if you provide your own column names ahead of time as the first row of your files and DataFrames that you want to ingest.

> The bytes of the data will be stored as a BlobField in the SQLite database file. Storing the data in the database not only (a) provides an entity that we can use to keep track of experiments and link relational data to but also (b) makes the data less mutable than keeping it in the open filesystem.

> You can choose whether or not you want to gzip compress the file when importing it with the `perform_gzip=bool` parameter. This compression not only enables you to store up to 90% more data on your local machine, but also helps overcome the maximum BlobField size of 2.147 GB. We handle the zipping and unzipping on the fly for you, so you don't even notice it.

> Optionally, `dtype`, as seen in [`pandas.DataFrame.astype(dtype)`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html), can be specified as either a single type for all columns, or as a dict that maps a specific type to each column name. This encodes features for analysis. We read NumPy into Pandas before persisting it, so `columns` and `dtype` are read directly by `pd.DataFrame()`.

> At this point, the project's support for Parquet is extremely minimal.

> If you leave `name` blank, it will default to a human-readble timestamp with the appropriate file extension (e.g. '2020_10_13-01_28_13_PM.tsv').

#### Fetch a `Dataset` with either **Pandas** or **NumPy**.

Supported in-memory formats include [NumPy Structured Array](https://numpy.org/doc/stable/user/basics.rec.html) and [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). 

```python
# Implicit IDs
df = dataset.to_pandas()
df.head()

arr = dataset.to_numpy()
arr[:4]

# Explicit IDs
df = aidb.Dataset.to_pandas(id=1)
df.head()

arr = aidb.Dataset.to_numpy(id=1)
arr[:4]
```

> For the sake of simplicity, we are reading into NumPy via Pandas. That way, if we want to revert to a simpler ndarray in the future, then we won't have to rewrite the function to read NumPy.

### 3. Select a `Label` column.

From a Dataset, pick a column that you want to train against/ predict. If you are planning on training an unsupervised model, then you don't need to do this.

```python
# Implicit IDs
label = dataset.make_label(column='species')

# Explicit IDs
label = aidb.Label.from_dataset(dataset_id=1, column='species')

label_col = label.column
```

Again, read a Label into memory with `to_pandas()` and `to_numpy()` methods.


### 4. Extract `Featureset` columns.

Creating a Featureset won't duplicate your data! It simply records the `columns` to be used in training. 

Here, we'll just exclude a Label column in preparation for supervised learning, but you can either exlcude or include any columns you see fit.

```python
# Implicit IDs
featureset = dataset.make_featureset(exclude_columns=[label_col])

# Explicit IDs
featureset = aidb.Featureset.from_dataset(
	dataset_id = 1
	, include_columns = None
	, exclude_columns = ['species']
)

>>> featureset.columns
['sepal length (cm)',
 'sepal width (cm)',
 'petal length (cm)',
 'petal width (cm)']

>>> featureset.columns_excluded
['species']
```

Again, read a Featureset into memory with `to_pandas()` and `to_numpy()` methods.

> The `include_columns` and `exclude_columns` parameters are provided to expedite column extraction:
> - If both `include_columns=None` and `exclude_columns=None` then all columns in the Dataset will be used.
> - If `exclude_columns=[...]` is specified, then all other columns will be included.
> - If `include_columns=[...]` is specified, then all other columns will be excluded. 
> - Remember, these parameters accept *[lists]*, not raw *strings*.


### 5. Generate `splits` of samples.

Divide the `Dataset` rows into `Splitsets` based on how you want to train, validate (optional), and test your models.

Again, creating a Splitset won't duplicate your data. It simply records the samples (aka rows) to be used in your train, validation, and test splits. 

```python
splitset_train75_test25 = featureset.make_splitset(label_name='species')

splitset_train70_test30 = featureset.make_splitset(
	label_name = 'species'
	, size_test = 0.30
)

splitset_train68_val12_test20 = featureset.make_splitset(
	label_name='species'
	, size_test = 0.20
	, size_validation = 0.12
)

splitset_unsupervised = featureset.make_splitset()
```

> Label-based stratification is used to ensure equally distributed label classes for both categorical and continuous data.

> The `size_test` and `size_validation` parameters are provided to expedite splitting samples:
> - If you leave `size_test=None`, it will default to `0.25` when a Label is provided.
> - You cannot specify `size_validation` without also specifying `size_test`.

Again, read a Splitset into memory with `to_pandas()` and `to_numpy()` methods. Note: this will return a `dict` of either data frames or arrays.

```python
>>> splitset_train68_val12_test20.sizes
{
	'train': {
		'percent': 0.68, 	
		'count': 102},

	'validation': {
		'percent': 0.12,
		'count': 18}, 

	'test':	{
		'percent': 0.2, 	
		'count': 30}
}

>>> splitset_train68_val12_test20.to_numpy()
{
	'train': {
		'features': <ndarray>,
		'labels': <ndarray>},

	'validation': {
		'features': <ndarray>,
		'labels': <ndarray>},

	'test': {
		'features': <ndarray>,
		'labels': <ndarray>}
}
```


### 6. Create an `Algorithm` aka model to fit to your splits.


### 7. Create combinations of `Hyperparamsets` for your algorithms.


### 8. Create a `Batch` of `Job`'s to keep track of training.

---

# PyPI Package

### Steps to Build & Upload

```bash
$ pyenv activate pydatasci
$ pip3 install --upgrade wheel twine
$ python3 setup.py sdist bdist_wheel
$ python3 -m twine upload --repository pypi dist/*
$ rm -r build dist pydatasci.egg-info
# proactively update the version number in setup.py next time
$ pip install --upgrade pydatasci; pip install --upgrade pydatasci
```