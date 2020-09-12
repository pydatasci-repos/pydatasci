name = "mldb"

import os, getpass, sqlite3
from os import path

import appdirs


def check_permissions(path:str):
	# learned that pip reads from appdirs on macos. so if they ran pip to install it they can r/w?
	readable = os.access(path, os.R_OK)
	writeable = os.access(path, os.R_OK)

	if not readable:
		print("\n=> Error - your operating system userID does not have permission to read from path:\n" + path + "\n")
	elif not writeable:
		print("\n=> Error - your operating system userID does not have permission to write to path:\n" + path + "\n")		
	elif not readable or not writeable:
		print("\n=> Fix - you can attempt to fix this by running `mldb.grant_appdirs_permissions()`.\n")
		return False
	elif readable and writeable:
		return True


def grant_appdirs_permissions():
	app_dir = appdirs.user_data_dir()
	command = "chmod +wr "
	full_command = command + '"' + app_dir + '"'

	try:
		sys_response = os.system(full_command)
	except:
		print("\nError - error failed to execute this system command: " + full_command +"\n")
	
	permissions = check_permissions(path=app_dir)
	if permissions == True:
		print("\nSuccess - operating system userID can read and write from path: " + app_dir + "\n")
	else:
		print("\nError - Failed to grant operating system userID permission to read and write from path: " + app_dir + "\n")


def create_db():
	# ToDo - Could let the user specify their own db name, for import tutorials.
	# ToDo - Need to know how to reference the default path globally.
	db_file_name = "pydatasci_mldb.sqlite3"
	app_dir = appdirs.user_data_dir()
	full_db_path = app_dir + db_file_name

	permissions = check_permissions(path=app_dir)

	if permissions == True:
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
				print("\n=> Error - failed to create database file at path:\n" + full_db_path)
				print("===================================\n")
				raise

			# verify its creation
			path_exists = path.exists(full_db_path)
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
		permissions = check_permissions(path=app_dir)

		if permissions == True:
			path_exists = path.exists(full_db_path)
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
