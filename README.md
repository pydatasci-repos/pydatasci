_pre-alpha_

---

# Mission
* **Automated**<br />_AIdb_ is an autoML tool that keeps track of the moving parts of machine learning (model tuning, feature selection, dataset splitting, and cross validation) so that data scientists can perform best practice ML without the coding overhead.<br /><br />

* **Local-first**<br />We empower non-cloud users (academic/ institute HPCs, private cloud companies, desktop hackers, or even remote server SSH'ers) with the same quality ML services as present in public clouds (e.g. SageMaker).<br /><br />

* **Integrated**<br />We don’t force your entire workflow into the confines of a GUI app or specific IDE because we integrate with your existing code.<br /><br />


### Functionality:
* Calculates and saves model metrics in a local SQLite file.
* Visually compare model metrics to find the best model.
* Queue for hypertuning jobs and batches.
* Treats cross-validated splits (k-fold) and validation sets (3rd split) as first-level citizens.
* Feature engineering to select the most informative columns.
* If you need to scale (data size, training time) just switch to `cloud_queue=True`.

---

# Installation:
Requires Python 3+. You will only need to do this the first time you use the package. Enter the following commands one-by-one and follow any instructions returned by the command prompt to resolve errors should they arise:

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
> _Installation Location Based on OS_<br />`appdir.user_data_dir('pydatasci')`:
> * Mac: <br />`/Users/Username/Library/Application Support/pydatasci`<br /><br />
> * Linux - Alpine and Ubuntu: <br />`/root/.local/share/pydatasci`<br /><br />
> * Windows: <br />`C:\Users\Username\AppData\Local\pydatasci`


### Deleting & Recreating the Database:
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

Let's get started.

```python
# comment
```

---

# PyPI Package

### Steps to Build & Upload:

```bash
$ pip3 install --upgrade wheel twine
$ python3 setup.py sdist bdist_wheel
$ python3 -m twine upload --repository pypi dist/*
$ rm -r build dist pydatasci.egg-info
# proactively update the version number in setup.py next time
$ pip install --upgrade pydatasci; pip install --upgrade pydatasci
```