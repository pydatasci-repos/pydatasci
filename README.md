# pydatasci

> Simplify the end-to-end workflow of machine learning.


### Updating the package:
```
$ rm -r build dist pydatasci.egg-info
# update version number in setup.py
$ python3 setup.py sdist bdist_wheel
$ python3 -m twine upload --repository pypi dist/*
$ pip3 install --upgrade pydatasci
```

### Installation:
This makes use of `appdirs` for an operating system agnostic location where a database file will be created.
```
$ pip3 install pydatasci
$ python3
>>> import pydatasci as pds
>>> pds.create_ml_database()
```
The path to the database will be set as global variable _ and used as a default argument with other functions, but you can override this argument if you need to.