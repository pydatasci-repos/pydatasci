![PyDataSci (wide)](logo_pds_wide.png)

---

# TLDR
```
pip install pydatasci

import pydatasci as pds
from pydatasci import aidb
```

# Value Proposition
PyDataSci's **_aidb_** is an open source, autoML tool that keeps track of the moving parts of machine learning so that data scientists can perform best practice ML without the coding overhead.

# Mission
* **Reproducibly Persistent & Embedded**<br />No more black boxes. No more screenshotting parameters and loss-accuracy graphs. A record of every: dataset, feature, sample, split, fold, parameter, run, model, and result - is automtically persisted in a lightweight, file-based database that is setup when you import the package. So hypertune to your heart's content, visually compare models, and know you've found the best one with the proof to back it up.<br /><br />

* **Local-First**<br />We empower non-cloud users: the academic/ institute HPCers, the private clouders, the remote server SSH'ers, and everyday desktop hackers - with the same quality ML tooling as present in public clouds (e.g. AWS SageMaker).<br /><br />

* **Code-Integrated**<br />We don’t disrupt the natural workflow of users by forcing them into the confines of a GUI app or specific IDE. Weave automated tracking into their existing code to work alongside the existing ecosystem of data science tools.<br /><br />

* **Scale-Em-If-You-Gottem**<br />Queue many hypertuning jobs locally, or run big jobs in parallel in the cloud by setting `cloud_queue = True`.<br /><br />


# Functionality
- Compress a dataset (csv, tsv, parquet) to be analyzed.
- Split your samples while treating validation sets (3rd split) and cross-folds (k-fold) as first-level citizens.
- Derive informative featuresets from that dataset using supervised and unsupervised methods.
- Queue hypertuning jobs and batches based on hyperparameter combinations.
- Evaluate and save the performance metrics of each model.
- Visually compare models to find the best one.
- Behind the scenes, stream rows from your datasets and use generators to keep a low memory footprint.
- Scale out to run cloud jobs in parallel by toggling `cloud_queue = True`.


# Painpoint Solved
At the time, I was deep in an unstable, remote Linux workspace trying to finish a meta-analysis of methods for interpreting neural network activation values as an alternative approach to predictions based on the traditional feedforward weighted sum. I was running so many variations of models from different versions of graph neural network algorithms, CNNs, LSTMs... the analysis was really starting to pile up. First I started taking screenshots of my loss-accuracy graphs and that worked fine for a while. Then I started taking screenshots of my hyper-params; I couldn't be bothered to write down every combination of parameters I was running and the performance metrics they spit out every time. But, then again, I hadn't generated confusion matrices to compare and I should really show record my feature importance ranking... and then the wheels really fell off when I started questioning if my `df` in-memory was really the `df` I thought it was last week. "Fuckkk," I said out loud, "I don't want to do all of that..." So I did what any good hacker would do and started a full blown project around it.

I had done the hardest part in figuring out the science, but this permuted world was just a mess when it came time to systematically prove it. It wasn't conducive to the scientific method. I had also been keeping an eye on other tools in the space. They seemed lacking in that they were either: cloud-only, dependent on an external database, the programming was too complex for data scientists/ statisticians/ researchers, the tech wasn't distributed properly, or they were just too proprietary/ walled garden/ biased toward corporate ecosystems.


# Community
*Much to automate there is. Simple it must be.* ML is a broad space with a lot of challenges to solve. Let us know if you want to get involved. We plan to host monthly dev jam sessions and data science lightning talks.

* **Data types to support:** tabular, time series, image, graph, audio, video, gaming.
* **Analysis types to support:** classification, regression, dimensionality reduction, feature engineering, recurrent, generative, reinforcement, NLP.

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
> We have attempted to support both Windows (`icacls` permissions and backslashes `C:\\`) as well as POSIX including Mac and Linux (`chmod letters` permissions and slashes `/`). Note: due to variations in the ordering of appdirs author and app directories in different OS', we do not make use of the appdirs `appauthor` directory, only the `appname` directory.
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

### 2. Add a `Dataset`.

Supported tabular file formats include: CSV, [TSV](https://stackoverflow.com/a/9652858/5739514), [Apache Parquet](https://parquet.apache.org/documentation/latest/). At this point, the project's support for Parquet is extremely minimal.

The bytes of the file will be stored as a BlobField in the SQLite database file. Storing the data in the database not only (a) provides an entity that we can use to keep track of experiments and link relational data to but also (b) makes the data less mutable than keeping it in the open filesystem.

```python
dataset = aidb.Dataset.from_file(
	path = 'iris.tsv' 
	,file_format = 'tsv'
	,name = 'tab-separated plants'
	,perform_gzip = True
)
```

> You can choose whether or not you want to gzip compress the file when importing it with the `perform_gzip=bool` parameter. This compression not only enables you to store up to 90% more data on your local machine, but also helps overcome the maximum BlobField size of 2.147 GB. We handle the zipping and unzipping on the fly for you, so you don't even notice it.

#### Fetch a `Dataset`.

Supported in-memory formats include: [NumPy Structured Array](https://numpy.org/doc/stable/user/basics.rec.html) and [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). 

#### Pandas & NumPy
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

## 2. Create a `Label` if you want to perform *supervised learning* (aka predict a specific column).

From a Dataset, pick a column that you want to train against/ predict. If you are planning on training an unsupervised model, then you don't need to do this.

```python
# Implicit IDs
label = dataset.make_label(column_name="species")

# Explicit IDs
label = aidb.Label.from_dataset(
	dataset_id = 1
	,column_name = 'species'
)

label_col = label.column
```

Read a Label into memory with `.to_pandas()` and `.to_numpy()`.


## 3. Derive a `Featureset` of columns from a Dataset.

Creating a Featureset won't duplicate your data! It simply records the `columns` to be used in training. The `include_columns` and `exclude_columns` parameters are provided for rapid splitting of data:

- If both `include_columns=None` and `exclude_columns=None` then all columns in the Dataset will be used.
- If `exclude_columns=[...]` is specified, then all other columns will be included.
- If `include_columns=[...]` is specified, then all other columns will be excluded. 
- Remember, these parameters accept *[lists]*, not raw *strings*.

Here, I'll just exclude a Label column in preparation for *supervised learning*. 

#### a) For *supervised learning*, be sure to pass in the `Label` you want to predict.

```python
# Implicit IDs
featureset = dataset.make_featureset(exclude_columns=[label_col])

# Explicit IDs
featureset = aidb.Featureset.from_dataset(
	dataset_id = 1
	, include_columns = None
	, exclude_columns = ["target"]
)
```

Read a Featureset into memory with `.to_pandas()` and `.to_numpy()`.

#### b) For *unsupervised learning* (aka studying variance within a `Dataset`), leave the `Label` blank.

Feature selection is about finding out which columns in your data are most informative. In performing feature engineering, a data scientist reduces the dimensionality of the data by determining the effect each feature has on the variance of the data. This makes for simpler models in the form of faster training and reduces overfitting by making the model more generalizable to future data.



## 4. Split the `Dataset` rows into `Splitsets` based on how you want to train, test, and validate your models.

- If you leave `size_test=None`, it will default to `0.25` when a Label is provided.
- You cannot specify `size_validation` without also specifying `size_test`.

#### a) **train-test** and **train-validate-test** splits.
```python
# Implicit
# ToDo Label by name
splitset_train75_test25 = featureset.make_splitset()
splitset_train70_test30 = featureset.make_splitset(size_test=0.30)
splitset_train68_val12_test20 = featureset.make_splitset(size_test=0.20,size_validation=0.12)
```

Read a Splitset into memory with `.to_pandas()` and `.to_numpy()`. This will return a `dict`.
```python
>>> splitset_train68_val12_test20.to_numpy()
{
	'train': {
		'features': <df or arr>,
		'labels': 	<df or arr>
	},
	'validation': {
		'features': <df or arr>,
		'labels': 	<df or arr>
	}	
	'test': {
		'features': <df or arr>,
		'labels': 	<df or arr>
	}
}

>>> splitset_train75_test25.sizes
{
	'train': 		{'percent': 0.68, 	'count': 102}
	'validation': 	{'percent': 0.12, 	'count': 18}, 
	'test': 		{'percent': 0.2, 	'count': 30}, 
}
```

#### b) k-fold sets containing **train-validate-test** splits.


## 5. Create an `Algorithm` aka model to fit to your splits.


## 6. Create combinations of `Hyperparamsets` for your algorithms.


## 7. Create a `Batch` of `Job`'s to keep track of training.

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