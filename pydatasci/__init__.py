import os, json

import appdirs


name = "pydatasci"

app_dir = appdirs.user_data_dir()
default_config_path = app_dir + "pydatasci_config.json"
default_db_path = app_dir + "pydatasci_db.sqlite3"


def check_path_permissions(path:str):
	# learned that pip reads from appdirs on macos. so if they ran pip to install it they can r/w?
	readable = os.access(path, os.R_OK)
	writeable = os.access(path, os.R_OK)

	if not readable:
		print("\n=> Error - your operating system userID does not have permission to read from path:\n" + path + "\n")
	elif not writeable:
		print("\n=> Error - your operating system userID does not have permission to write to path:\n" + path + "\n")		
	elif not readable or not writeable:
		print("\n=> Fix - you can attempt to fix this by running `pds.grant_appdirs_permissions()`.\n")
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
		print("\n=> Error - error failed to execute this system command: " + full_command +"\n")
	
	permissions = check_path_permissions(path=app_dir)
	if permissions == True:
		print("\n=> Success - operating system userID can read and write from path: " + app_dir + "\n")
	else:
		print("\n=> Error - Failed to grant operating system userID permission to read and write from path: " + app_dir + "\n")


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
		permissions = check_path_permissions(path=app_dir)
		if permissions:

			pds_config = {
				"config_path": default_config_path,
				"db_path": default_db_path,
			}
			
			try:
				with open(default_config_path, 'w') as pds_config_file:
					json.dump(pds_config, pds_config_file)
			except:
				print("\n=> Error - failed to create config file at path:\n" + default_config_path)
				print("===================================\n")
				raise
			print("\n=> Success - created config file for settings at path:\n" + default_config_path + "\n")
	else:
		print("\n=> Warning - skipping as config file already exists at path: " + default_config_path + "\n")


def delete_config(confirm:bool):
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


pds_config = get_config()
