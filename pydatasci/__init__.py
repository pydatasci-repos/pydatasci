import os, json

import appdirs


name = "pydatasci"

app_dir_no_trailing_slash = appdirs.user_data_dir("pydatasci")
# Adds trailing slash or backslashes depending on OS.
app_dir = os.path.join(app_dir_no_trailing_slash, '')
default_config_path = app_dir + "config.json"
default_db_path = app_dir + "aidb.sqlite3"


def check_exists_folder():
	# If Windows does not have permission to read the folder, it will fail when trailing backslashes \\ provided.
	app_dir_exists = os.path.exists(app_dir_no_trailing_slash)
	if app_dir_exists:
		print("\n=> Success - the following file path already exists on your system:\n" + app_dir + "\n")
		return True
	else:
		print("\n=> Info - it appears the following folder does not exist on your system:\n" + app_dir + "\n")
		print("\n=> Fix - you can attempt to fix this by running `pds.create_folder()`.\n")
		return False


def create_folder():
	app_dir_exists = check_exists_folder()
	if app_dir_exists:
		print("\n=> Info - skipping folder creation as folder already exists at file path:\n" + app_dir + "\n")
	else:
		# ToDo - windows support.
		try:
			if os.name == 'nt':
				# Windows: backslashes \ and double backslashes \\
				command = 'mkdir ' + app_dir
				os.system(command)
			else:
				# posix (mac and linux)
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
		# Windows `os.access()` always returning True even when I have verify permissions are in fact denied.
		if os.name == 'nt':
			# Test write.
			file_name = "pds_test_permissions.txt"
			
			try:
				cmd_file_create = 'echo "test" >> ' + app_dir + file_name
				write_response = os.system(cmd_file_create)
			except:
				print("\n=> Error - your operating system user does not have permission to write to file path:\n" + app_dir + "\n")
				print("\n=> Fix - you can attempt to fix this by running `pds.grant_permissions_folder()`.\n")
				return False

			if write_response != 0:
				print("\n=> Error - your operating system user does not have permission to write to file path:\n" + app_dir + "\n")
				print("\n=> Fix - you can attempt to fix this by running `pds.grant_permissions_folder()`.\n")
				return False
			else:
				# Test read.
				try:
					read_response = os.system("type " + app_dir + file_name)
				except:
					print("\n=> Error - your operating system user does not have permission to read from file path:\n" + app_dir + "\n")
					print("\n=> Fix - you can attempt to fix this by running `pds.grant_permissions_folder()`.\n")
					return False

				if read_response != 0:
					print("\n=> Error - your operating system user does not have permission to read from file path:\n" + app_dir + "\n")
					print("\n=> Fix - you can attempt to fix this by running `pds.grant_permissions_folder()`.\n")
					return False
				else:
					cmd_file_delete = "erase " + app_dir + file_name
					os.system(cmd_file_delete)
					print("\n=> Success - your operating system user can read from and write to file path:\n" + app_dir + "\n")
					return True

		else:
			# posix
			# https://www.geeksforgeeks.org/python-os-access-method/
			readable = os.access(app_dir, os.R_OK)
			writeable = os.access(app_dir, os.W_OK)

			if readable and writeable:
				print("\n=> Success - your operating system user can read from and write to file path:\n" + app_dir + "\n")
				return True
			else:
				if not readable:
					print("\n=> Error - your operating system user does not have permission to read from file path:\n" + app_dir + "\n")
				if not writeable:
					print("\n=> Error - your operating system user does not have permission to write to file path:\n" + app_dir + "\n")
				if not readable or not writeable:
					print("\n=> Fix - you can attempt to fix this by running `pds.grant_permissions_folder()`.\n")
					return False
	else:
		return False


def grant_permissions_folder():
	permissions = check_permissions_folder()
	if permissions:
		print("\n=> Info - skipping as you already have permissions to read from and write to file path:\n" + app_dir + "\n")
	else:
		try:
			if os.name == 'nt':
				# Windows ICACLS permissions: https://www.educative.io/edpresso/what-is-chmod-in-windows
				# Works in Windows Command Prompt and `os.system()`, but not PowerShell.
				# Does not work with trailing backslashes \\
				command = 'icacls "' + app_dir_no_trailing_slash + '" /grant users:(F) /c'
				os.system(command)
			else:
				# posix
				command = 'chmod +wr ' + '"' + app_dir + '"'
				os.system(command)
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
		print("\n=> Welcome to PyDataSci.\nTo get started, run `pds.create_folder()` followed by `pds.create_config()` in Python shell.\n")


def create_config():
	#check if folder exists
	folder_exists = check_exists_folder()
	if folder_exists:
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
			print("\n=> Info - skipping as config file already exists at path:\n" + default_config_path + "\n")


def delete_config(confirm:bool=False):
	pds_config = get_config()
	if pds_config is None:
		print("\n=> Info - skipping as there is no config file to delete.\n")
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
			print("\n=> Info - skipping deletion because `confirm` arg not set to boolean `True`.\n")


def update_config(kv:dict):
	pds_config = get_config()
	if pds_config is None:
		print("\n=> Info - there is no config file to update.\n")
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

# This runs at startup and triggers the welcome message instructions if configuration has not taken place yet.
pds_config = get_config()
