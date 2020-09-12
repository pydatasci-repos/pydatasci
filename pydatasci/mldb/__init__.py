name = "mldb"

import os, sqlite3
from os import path

import appdirs

# add a whoami for helping troubleshoot and checking permissions?
# make this printing a func()
# print("\n=> Check that the operating system user that you are logged in as\n  has permission to create files at path:\n" + app_dir)
# is $USER universal?

def create_db():
	# Could let the user specify their own db name, for import tutorials.
	# Need to know how to reference the default path globally.
	db_file_name = "pydatasci_mldb.sqlite3"
	full_db_path = appdirs.user_data_dir(db_file_name)
	app_dir = appdirs.user_data_dir()

	# check if file exists already
	path_exists = path.exists(full_db_path)
	if (path_exists == True):
		print("\n=> Warning - skipping creation as a database already exists at path:\n" + full_db_path + "\n")
	else:
		# attempt to create it
		try:
			conn = sqlite3.connect(full_db_path)
			del conn
		except:
			print("\n=> Warning - failed to create database file at path:\n" + full_db_path)
			print("===================================\n")
			raise

		# verify its creation
		path_exists = path.exists(full_db_path)
		if (path_exists == True):
			print("\n=> Successfully created database for machine learning metrics at path:\n" + full_db_path + "\n")
			
			print("=> Returning string of the path so that it can be set as a variable:")
			return full_db_path
		else:
			print("\n=> Warning - failed to create database at path:\n" + full_db_path + "\n")

	# Need to create the tables

def delete_db(confirm:bool):
	# Need to know how to reference the default path globally.
	db_file_name = "pydatasci_mldb.sqlite3"
	full_db_path = appdirs.user_data_dir(db_file_name)

	if (confirm == True):		
		path_exists = path.exists(full_db_path)
		if (path_exists == True):
			try:
				os.remove(full_db_path)
			except:
				print("\n=> Warning - failed to delete database at path:\n" + full_db_path)
				print("===================================")
				raise
			print("\n=> Successfully deleted database at path:\n" + full_db_path + "\n")	
		else:
			print("\n=> Warning - there is no file to delete at path:\n" + full_db_path + "\n")

	else:
		print("\n=> Warning - skipping deletion because `confirm` arg not set to boolean `True`.\n")


def say():
	print("\nA little something into the camera.\n")
