name = "mldb"

import os, getpass, sqlite3

import appdirs

from pydatasci import get_config


def get_db_path():
	pds_config = get_config()
	if pds_config is None:
		print("\n Welcome - configuration not set, run `pds.create_config()` in Python shell.\n")
	else:
		db_path = pds_config['db_path']
	
	return db_path

def create_db():
	# ToDo - Could let the user specify their own db name, for import tutorials.
	db_path = get_db_path()
	db_exists = os.path.exists(db_path)
	if db_exists:
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
		print("\n=> Success - created database for machine learning metrics at path:\n" + db_path + "\n")

	# ToDo - Need to create the tables


def delete_db(confirm:bool):
	if confirm:
		db_path = get_db_path()
		db_exists = os.path.exists(db_path)
		if db_exists:
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


# get should contain the conn? or conn as it's own? or conn with config?
