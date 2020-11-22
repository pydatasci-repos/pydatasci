name = "examples"

import pkg_resources #importlib.resources was not working on Google Collab.

import pandas as pd

from pydatasci import aidb


def get_data_as_pandas(file_name:str):
	short_path = f"data/{file_name}"
	full_path = pkg_resources.resource_filename('pydatasci', short_path)

	if 'tsv' in file_name:
		separator = '\t'
	else:
		separator = None

	df = pd.read_csv(full_path, sep=separator)
	return df