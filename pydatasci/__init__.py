name = "pydatasci"

import os, sqlite3
from os import path

import appdirs


def create_ml_database():
	app_dir = appdirs.user_data_dir(name)
	# Assuming I don't have sudo permissions to create a directory.
	db_name = '_mldb.sqlite3'
	full_db_path = app_dir + db_name

	print("--> Attempting to create database at path: " + full_db_path)
	path_exists = path.exists(full_db_path)

	if (path_exists == True):
		print("--> Error: skipping creation as a database already exists at path: " + full_db_path)
	else:
		conn = sqlite3.connect(full_db_path)
		del conn
		
		path_exists = path.exists(full_db_path)
		if (path_exists == True):
			print("--> Successfully created database for machine learning metrics at path: " + full_db_path)	
		else:
			print("--> Failed to create database at path: " + full_db_path)
	
	# Need to create the tables
	# Need to be able to reference this path in the rest of the package whenever I need to make a connection.

	return full_db_path

"""
# Need to know how to reference the path globally.
def delete_ml_database(confirm):
	if (confirm == True):
		os.remove(full_db_path)
		print("--> Successfully deleted database at path: " + full_db_path)	
	else:
		print("--> Error: skipping deletion because `confirm` arg not set to `True`.")
"""

def say():
	print("A little something into the camera.")
