name = "mldb"

import os, getpass, sqlite3

import appdirs

from pydatasci import get_config
pds_config = get_config()
db_path = pds_config['db_path']


def create_db():
	# ToDo - Could let the user specify their own db name, for import tutorials.
	path_exists = os.path.exists(db_path)
	if path_exists:
		print("\n=> Warning - skipping creation as a database already exists at path:\n" + db_path + "\n")
	else:
		# attempt to create it
		try:
			conn = sqlite3.connect(db_path)
			del conn
		except:
			print("\n=> Error - failed to create database file at path:\n" + db_path)
			print("===================================\n")
			raise

		# verify its creation
		path_exists = os.path.exists(db_path)
		if (path_exists == True):
			print("\n=> Success - created database for machine learning metrics at path:\n" + db_path + "\n")
			
			print("=> Success - Returning string of the path so that it can be set as a variable:")
			return db_path
		else:
			print("\n=> Error - failed to create database at path:\n" + db_path + "\n")

	# ToDo - Need to create the tables


def delete_db(confirm:bool):
	# Need to know how to reference the default path globally.
	if confirm:
		path_exists = os.path.exists(db_path)
		if (path_exists == True):
			try:
				os.remove(db_path)
			except:
				print("\n=> Error - failed to delete database at path:\n" + db_path)
				print("===================================")
				raise
			print("\n=> Success - deleted database at path:\n" + db_path + "\n")	
		else:
			print("\n=> Warning - there is no file to delete at path:\n" + db_path + "\n")
	else:
		print("\n=> Warning - skipping deletion because `confirm` arg not set to boolean `True`.\n")


def say():
	print("\nA little something into the camera.\n")
