name = "mldb"

import os, getpass, sqlite3

import appdirs

from pydatasci import pds_config
print(pds_config)

def create_db():
	# ToDo - Could let the user specify their own db name, for import tutorials.
	# ToDo - Need to know how to reference the default path globally.
	db_file_name = "pydatasci_mldb.sqlite3"
	app_dir = appdirs.user_data_dir()
	full_db_path = app_dir + db_file_name

	permissions = check_path_permissions(path=app_dir)

	if permissions:
		# check if file exists already
		path_exists = os.path.exists(full_db_path)
		if path_exists:
			print("\n=> Warning - skipping creation as a database already exists at path:\n" + full_db_path + "\n")
		else:
			# attempt to create it
			try:
				conn = sqlite3.connect(full_db_path)
				del conn
			except:
				print("\n=> Error - failed to create database file at path:\n" + full_db_path)
				print("===================================\n")
				raise

			# verify its creation
			path_exists = os.path.exists(full_db_path)
			if (path_exists == True):
				print("\n=> Success - created database for machine learning metrics at path:\n" + full_db_path + "\n")
				
				print("=> Success - Returning string of the path so that it can be set as a variable:")
				return full_db_path
			else:
				print("\n=> Error - failed to create database at path:\n" + full_db_path + "\n")
	
	# ToDo - Need to create the tables


def delete_db(confirm:bool):
	# Could let the user specify their own db name, for import tutorials.
	# Need to know how to reference the default path globally.
	db_file_name = "pydatasci_mldb.sqlite3"
	app_dir = appdirs.user_data_dir()
	full_db_path = app_dir + db_file_name

	if (confirm == True):		
		permissions = check_path_permissions(path=app_dir)

		if permissions == True:
			path_exists = os.path.exists(full_db_path)
			if (path_exists == True):
				try:
					os.remove(full_db_path)
				except:
					print("\n=> Error - failed to delete database at path:\n" + full_db_path)
					print("===================================")
					raise
				print("\n=> Success - deleted database at path:\n" + full_db_path + "\n")	
			else:
				print("\n=> Warning - there is no file to delete at path:\n" + full_db_path + "\n")

	else:
		print("\n=> Warning - skipping deletion because `confirm` arg not set to boolean `True`.\n")


def say():
	print("\nA little something into the camera.\n")
