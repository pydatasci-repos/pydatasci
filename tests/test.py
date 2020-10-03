import pydatasci as pds; from pydatasci import aidb; import os, sqlite3, io, gzip, pandas as pd

from importlib import reload; aidb.delete_db(True); reload(aidb); aidb.create_db()



d = aidb.Dataset.create_from_file('iris.csv','csv')
l = aidb.Label.create_from_dataset(1,'target')


f2 = aidb.Featureset.create_all_columns(dataset_id=d.id, label_id=l.id)
f2.tags
f2.supervision
f2.label




features_train, features_test, labels_train, labels_test = aidb.Foldset.create_from_featureset(f2.id)







f1 = aidb.Featureset.create_from_dataset_columns(dataset_id=d.id, label_id=l.id, columns=['petal width (cm)'])
f1.tags
f1.supervision
f1.label




f3 = aidb.Featureset.create_from_dataset_columns(dataset_id=d.id, columns=['petal length (cm)', 'petal width (cm)'])
f3.tags
f3.supervision
f3.label



f4 = aidb.Featureset.create_all_columns(dataset_id=d.id)
f4.tags
f4.supervision
f4.label


l.featuresets





