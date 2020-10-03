import pydatasci as pds; from pydatasci import aidb; import os, sqlite3, io, gzip, pandas as pd

from importlib import reload; aidb.delete_db(True); reload(aidb); aidb.create_db()


cols = ['sepal width (cm)', 'sepal length (cm)']



d1 = aidb.Dataset.create_from_file('iris.csv','csv')
d2 = aidb.Dataset.create_from_file('iris.tsv','tsv')
d3 = aidb.Dataset.create_from_file('iris.parquet','parquet')
d4 = aidb.Dataset.create_from_file('iris.csv','csv', perform_gzip=False)
d5 = aidb.Dataset.create_from_file('iris.tsv','tsv', perform_gzip=False)
d6 = aidb.Dataset.create_from_file('iris.parquet','parquet', perform_gzip=False)



df7 = aidb.Dataset.read_to_pandas(id = 1)
df8 = aidb.Dataset.read_to_pandas(id = 2)
df9 = aidb.Dataset.read_to_pandas(id = 3)
df10 = aidb.Dataset.read_to_pandas(id = 4)
df11 = aidb.Dataset.read_to_pandas(id = 5)
df12 = aidb.Dataset.read_to_pandas(id = 6)

df19 = aidb.Dataset.read_to_numpy(id = 1)
df20 = aidb.Dataset.read_to_numpy(id = 2)
df21 = aidb.Dataset.read_to_numpy(id = 3)
df22 = aidb.Dataset.read_to_numpy(id = 4)
df23 = aidb.Dataset.read_to_numpy(id = 5)
df24 = aidb.Dataset.read_to_numpy(id = 6)


df7.head()
df8.head()
df9.head()
df10.head()
df11.head()
df12.head()

df19[:4]
df20[:4]
df21[:4]
df22[:4]
df23[:4]
df24[:4]




df1 = aidb.Dataset.read_to_pandas(id = 1, columns=cols) 
df2 = aidb.Dataset.read_to_pandas(id = 2, columns=cols) 
df3 = aidb.Dataset.read_to_pandas(id = 3, columns=cols)
df4 = aidb.Dataset.read_to_pandas(id = 4, columns=cols)
df5 = aidb.Dataset.read_to_pandas(id = 5, columns=cols)
df6 = aidb.Dataset.read_to_pandas(id = 6, columns=cols) 

df13 = aidb.Dataset.read_to_numpy(id = 1, columns=cols) 
df14 = aidb.Dataset.read_to_numpy(id = 2, columns=cols) 
df15 = aidb.Dataset.read_to_numpy(id = 3, columns=cols)
df16 = aidb.Dataset.read_to_numpy(id = 4, columns=cols)
df17 = aidb.Dataset.read_to_numpy(id = 5, columns=cols)
df18 = aidb.Dataset.read_to_numpy(id = 6, columns=cols) 




df1.head()
df2.head()
df3.head()
df4.head()
df5.head()
df6.head()

df13[:4]
df14[:4]
df15[:4]
df16[:4]
df17[:4]
df18[:4]
