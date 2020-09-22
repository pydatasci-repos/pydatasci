import os, json

import appdirs


name = "pydatasci"

app_dir_free = appdirs.user_data_dir()
# need to make sure there's a trailing slash
app_dir = os.path.join(app_dir_free, '')
default_config_path = app_dir + "pydatasci_config.json"
default_db_path = app_dir + "pydatasci_db.sqlite3"


def check_exists_folder():
	app_dir_exists = os.path.exists(app_dir)
	if app_dir_exists:
		print("\n=> Success - the following file path already exists on your system:\n" + app_dir + "\n")
	else:
		print("\n=> Error - the following file path does not exist on your system:\n" + app_dir + "\n")
		print("\n=> Fix - you can attempt to fix this by running `pds.create_folder()`.\n")

	return app_dir_exists


def create_folder():
	app_dir_exists = check_exists_folder()
	if app_dir_exists:
		print("\n=> Warning - skipping folder creation as folder already exists at file path:\n" + app_dir + "\n")
	else:
		# ToDo - windows support.
		try:
			if os.name == 'nt':
				# Windows: remember backslashes \
				command = 'mkdir ' + app_dir
				os.system(command)
			else:
				# posix
				command = 'mkdir -p "' + app_dir + '"'
				os.system(command)
		except:
			print("\n=> Error - error failed to execute this system command:\n" + command)
			print("===================================\n")
			raise
		print("\n=> Success - created folder at file path:\n" + app_dir + "\n")
		print("\n=> Fix - now try running `pds.create_config()` again.\n")


def check_permissions_folder():
	app_dir_exists = check_exists_folder()
	if app_dir_exists:
		# https://www.geeksforgeeks.org/python-os-access-method/
		readable = os.access(app_dir, os.R_OK)
		writeable = os.access(app_dir, os.W_OK)

		if readable and writeable:
			print("\n=> Success - your operating system userID can read and write to file path:\n" + app_dir + "\n")
			return True
		else:
			if not readable:
				print("\n=> Error - your operating system userID does not have permission to read from file path:\n" + app_dir + "\n")
			if not writeable:
				print("\n=> Error - your operating system userID does not have permission to write to file path:\n" + app_dir + "\n")
			if not readable or not writeable:
				print("\n=> Fix - you can attempt to fix this by running `pds.grant_permissions_folder()`.\n")
				return False
	else:
		return False


def grant_permissions_folder():
	try:
		if os.name == 'nt':
			# Windows icalcs permissions: https://www.educative.io/edpresso/what-is-chmod-in-windows
			# ToDo - test on a Windows machine and mess with permissions before and after db file creation.
			command = 'icacls "' + app_dir + '" /grant users:(F) /t /c'
			sys_response = os.system(command)
		else:
			# posix
			command = 'chmod +wr ' + '"' + app_dir + '"'
			sys_response = os.system(command)
	except:
		print("\n=> Error - error failed to execute this system command:\n" + command)
		print("===================================\n")
		raise
	
	permissions = check_permissions_folder()
	if permissions:
		print("\n=> Success - granted system permissions to read and write from file path:\n" + app_dir + "\n")
	else:
		print("\n=> Error - failed to grant system permissions to read and write from file path:\n" + app_dir + "\n")


def get_config():
	pds_config_exists = os.path.exists(default_config_path)
	if pds_config_exists:
		with open(default_config_path, 'r') as pds_config_file:
			pds_config = json.load(pds_config_file)
			return pds_config
	else: 
		print("\n=> Welcome to pydatasci. Configuration not set, run `pds.create_config()` in Python shell.\n")


def create_config():
	config_exists = os.path.exists(default_config_path)
	if not config_exists:
		pds_config = {
			"config_path": default_config_path,
			"db_path": default_db_path,
		}
		
		try:
			with open(default_config_path, 'w') as pds_config_file:
				json.dump(pds_config, pds_config_file)
		except:
			print("\n=> Error - failed to create config file at path:\n" + default_config_path)
			print("\n=> Fix - you can attempt to fix this by running `pds.check_permissions_folder()`.\n")
			print("===================================\n")
			raise
		print("\n=> Success - created config file for settings at path:\n" + default_config_path + "\n")
	else:
		print("\n=> Warning - skipping as config file already exists at path: " + default_config_path + "\n")


def delete_config(confirm:bool=False):
	pds_config = get_config()
	if pds_config is None:
		print("\n=> Warning - skipping as there is no config file to delete.\n")
	else:
		if confirm:
			config_path = pds_config['config_path']
			try:
				os.remove(config_path)
			except:
				print("\n=> Error - failed to delete config file at path:\n" + config_path)
				print("===================================\n")
				raise
			print("\n=> Success - deleted config file at path:\n" + config_path + "\n")		
		else:
			print("\n=> Warning - skipping deletion because `confirm` arg not set to boolean `True`.\n")


def update_config(kv:dict):
	pds_config = get_config()
	if pds_config is None:
		print("\n=> Warning - there is no config file to update.\n")
	else:
		for k, v in kv.items():
			pds_config[k] = v		
		config_path = pds_config['config_path']
		
		try:
			with open(config_path, 'w') as pds_config_file:
				json.dump(pds_config, pds_config_file)
		except:
			print("\n=> Error - failed to update config file at path:\n" + config_path)
			print("===================================\n")
			raise
		print("\n=> Success - updated configuration settings:\n" + str(pds_config) + "\n")

# This runs at startup and triggers the welcome message/ instruction if it is not configured.
pds_config = get_config()
