# pydatasci

> Simplify the end-to-end workflow of machine learning.


### Updating the package:
```
# update version number in setup.py
$ python3 setup.py sdist bdist_wheel
$ python3 -m twine upload --repository pypi dist/*
$ pip3 install --upgrade pydatasci
$ rm -r build dist pydatasci.egg-info
```

### Installation:
This makes use of `appdirs` for an operating system agnostic location where a database file will be created.
```
$ pip3 install --upgrade pydatasci
$ python3
>>> from pydatasci import mldb as mldb
>>> mldb.create_db()
```
The path to the database will be set as global variable _ and used as a default argument with other functions, but you can override this argument if you need to.