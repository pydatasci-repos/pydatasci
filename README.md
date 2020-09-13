# pydatasci

> Simplify the end-to-end workflow of machine learning.


### Updating PyPI Package:
```
$ python3 setup.py sdist bdist_wheel
$ python3 -m twine upload --repository pypi dist/*
$ pip3 install --upgrade pydatasci; pip3 install --upgrade pydatasci
$ rm -r build dist pydatasci.egg-info
# proactively update the version number in setup.py next time
```

### First Time Installation:
This library makes use of `appdirs` for an operating system agnostic location where configuration and database files will be created to store settings and data science metrics. This process also ensures that you have the permissions needed to read/ write files in that location. 

Enter the following commands one by one:
```
$ pip3 install --upgrade pydatasci
$ python3

>>> import pydatasci as pds
>>> pds.check_path_permissions(pds.app_dir)
>>> pds.create_config()

>>> from pydatasci import mldb
>>> mldb.create_db()
```


ToDo - The path to the database will be set as global variable _ and used as a default argument with other functions, but you can override this argument if you need to.