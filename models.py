import numpy as np

class LinearModel(object):

	def __init__(self, training_data, polynomial_degree=1):
		self.training_data = training_data
		self.raw_training_inputs, self.training_outputs = training_data
		self.polynomial_degree = polynomial_degree
		self.training_inputs = self.process_raw_inputs(self.raw_training_inputs)
		number_of_inputs = self.training_inputs.shape[1]
		self.initialise_parameters(number_of_inputs)

	def describe(self):
		print 'Model Details'
		print '- Polynomial degree:', self.polynomial_degree
		print '- Parameters:', self.parameters
		number_of_datapoints, number_of_raw_inputs = self.raw_training_inputs.shape
		number_of_inputs = self.training_inputs.shape[1]
		print 'Training Data'
		print '- Number of datapoints:', number_of_datapoints
		print '- Number of raw features:', number_of_raw_inputs
		print '- Total number of features:', number_of_inputs

	def process_raw_inputs(self, raw_inputs):
		num_rows = raw_inputs.shape[0]
		column_of_constants = np.ones((num_rows, 1))
		inputs_with_constant_term = np.hstack((column_of_constants, raw_inputs))
		processed_inputs = generate_polynomial_features(inputs_with_constant_term, self.polynomial_degree)
		return processed_inputs

	def model(self, features, parameters):
		return np.dot(features, parameters)

	def predict(self, inputs):
		features = generate_polynomial_features(inputs, polynomial_degree)
		return self.model(features, self.parameters)

	def train(self, cost_class, optimiser_class, optimiser_options):
		cost = cost_class(self.training_inputs, self.training_outputs, self.model)
		optimiser = optimiser_class(optimiser_options)
		self.parameter_history, self.cost_history = optimiser.optimise(cost, self.parameters)
		self.parameters = self.parameter_history[-1]

	def initialise_parameters(self, number_of_parameters):
		self.parameters = np.zeros((number_of_parameters, 1))

	def plot_cost_history(self):
		pass


def generate_polynomial_features(features, degree):
	return features
