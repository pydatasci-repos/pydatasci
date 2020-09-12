import json

import appdirs

name = "pydatasci"

app_dir = appdirs.user_data_dir()


config_file_path = app_dir + 'pydatasci_config.json'

# should really make this a configure() step.
with open(config_file_path, 'r') as pydatasci_config_file:
	pydatasci_config = json.load(pydatasci_config_file)

pydatasci_config['app_dir']=app_dir


with open(config_file_path, 'w') as pydatasci_config_file:
	json.dump(pydatasci_config, pydatasci_config_file)


# ToDo - import this pydatasci_config vartiable from MLdb script.