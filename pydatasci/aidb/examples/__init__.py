name = "examples"

import pkg_resources #importlib.resources was not working on Google Collab.

from pydatasci.aidb import *


def get_file_path(file_name:str):
	short_path = f"data/{file_name}"
	full_path = pkg_resources.resource_filename('pydatasci', short_path)
	return full_path


def get_file_as_pandas(file_name:str):
	file_path = get_file_path(file_name)

	if 'tsv' in file_name:
		separator = '\t'
	else:
		separator = None

	df = pd.read_csv(file_path, sep=separator)
	return df


"""
Due to `pickle` not handling nested functions, 
These dummy model functions must be defined outside of the function that accesses them.
For example when creating an `def example_method()... Algorithm.function_model_build`
"""
def function_model_build(**hyperparameters):
		model = Sequential()
		model.add(Dense(13, input_shape=(4,), activation='relu', kernel_initializer='he_uniform'))
		model.add(Dropout(0.2))
		model.add(Dense(hyperparameters['l2_neuron_count'], activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(3, activation='softmax', name='output'))

		opt = keras.optimizers.Adamax(hyperparameters['learning_rate'])

		model.compile(
			loss = 'categorical_crossentropy'
			, optimizer = opt
			, metrics = ['accuracy']
		)
		return model

def function_model_train(model, samples_train, samples_evaluate, **hyperparameters):
	model.fit(
		samples_train["features"]
		, samples_train["labels"]
		, validation_data = (
			samples_evaluate["features"]
			, samples_evaluate["labels"]
		)
		, verbose = 0
		, batch_size = 3
		, epochs = hyperparameters['epochs']
		, callbacks=[History()]
	)
	return model

def function_model_predict(model, samples_predict):
	probabilities = model.predict(samples_predict['features'])
	predictions = np.argmax(probabilities, axis=-1)
	return predictions, probabilities

def function_model_loss(model, samples_evaluate):
	loss, _ = model.evaluate(samples_evaluate['features'], samples_evaluate['labels'], verbose=0)
	return loss


def make_batch_supervised_multiclass():
	#is_folded:bool
	file_path = get_file_path('iris_10x.tsv')

	dataset = Dataset.from_file(
		path = file_path
		, file_format = 'tsv'
		, name = 'tab-separated plants duplicated 10 times.'
		, perform_gzip = True
		, dtype = None
	)
	
	label_column = 'species'
	label = dataset.make_label(columns=[label_column])

	featureset = dataset.make_featureset(exclude_columns=[label_column])

	splitset = featureset.make_splitset(
		label_id = label.id
		, size_test = 0.20
		, size_validation = 0.12
	)

	encoder_features = StandardScaler()
	encoder_labels = OneHotEncoder(sparse=False)

	preprocess = splitset.make_preprocess(
		description = "standard scaling on features"
		, encoder_features = encoder_features
		, encoder_labels = encoder_labels
	)

	hyperparameters = {
		"l2_neuron_count": [9, 13]
		, "learning_rate": [0.03, 0.05]
		, "epochs": [50, 100]
	}

	algorithm = Algorithm.create(
		library = "Keras"
		, analysis_type = "classification_multi"
		, function_model_build = function_model_build
		, function_model_train = function_model_train
		, function_model_predict = function_model_predict
		, function_model_loss = function_model_loss
	)

	hyperparamset = algorithm.make_hyperparamset(
		preprocess_id = preprocess.id
		, hyperparameters = hyperparameters
	)

	batch = algorithm.make_batch(
		splitset_id = splitset.id
		, foldset_id = None
		, hyperparamset_id = hyperparamset.id
	)
	return batch