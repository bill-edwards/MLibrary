import numpy as np
import matplotlib.pyplot as plt
import utils.sigmoid as sigmoid

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
		plt.plot(self.cost_history)
		plt.show()


class BinaryLogisticClassifier(LinearModel):

	def model(self, features, parameters):
		return sigmoid(np.dot(features, parameters))

	def predict(self, inputs):
		return 1 if (np.dot(features, parameters) > 0) else 0


# Example with two raw features, generating cubic features.
# [1, x, y] # features
# ->
# [[1], [x], [y]] # products before loop
# [[1, x, y], [x^2, xy], [y^2]] # products after 1st loop
# [[1, x, y, x^2, xy, y^2], [x^3, x^2y, xy^2], [y^3]] # products after 2nd loop
# ->
# [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3] # return value

def generate_polynomial_features(features, degree):
	return features
	# num_features = features.size
	# products = [np.array([feature]) for feature in features] # Create a list of 1-element arrays, one for each feature.
	# for power in range(1, degree):
	# 	for i in range(num_features):
	# 		ith_products = []
	# 		for row in products[i:]: # The ith feature multiplies its own products, and all products deriving from features j > i.
	# 			ith_products.append(features[i]*row)
	# 		products[i] = np.hstack(ith_products)
	# return np.hstack(products)

