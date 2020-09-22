# pydatasci

> Simplify the end-to-end workflow of machine learning.


### Updating PyPI Package:
```
$ pip3 install --upgrade wheel twine
$ python3 setup.py sdist bdist_wheel
$ python3 -m twine upload --repository pypi dist/*
$ rm -r build dist pydatasci.egg-info
# proactively update the version number in setup.py next time
$ pip3 install --upgrade pydatasci; pip3 install --upgrade pydatasci
```


### First Time Installation:
This library makes use of `appdirs` for an operating system agnostic location where configuration and database files will be created to store settings and data science metrics. This process also ensures that you have the permissions needed to read from/ write to that folder, and that the folder exists in the first place. 

Enter the following commands one by one and follow any instructions returned by the command prompt to resolve errors:
```
$ pip3 install --upgrade pydatasci
$ python3

>>> import pydatasci as pds
>>> pds.create_config()
>>> from pydatasci import aidb
>>> aidb.create_db()
```


## Deleting & Recreating the Database:
When deleting the database, you need to either reload the aidb module or restart the Python shell before you can attempt to recreate the database.
```
aidb.delete_db()
from importlib import reload
reload(aidb)
create_db()
```