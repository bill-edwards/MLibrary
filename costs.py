import numpy as np

class LeastSquares(object):

	def __init__(self, training_inputs, training_outputs, model_function):
		self.training_inputs = training_inputs
		self.training_outputs = training_outputs
		self.model_function = model_function

	def cost_function(self, parameters):
		difference_vector = self.training_outputs - self.model_function(self.training_inputs, parameters)
		cost = np.transpose(difference_vector).dot(difference_vector).reshape(())
		return cost

	def first_derivative(self, parameters):
		difference_vector = self.training_outputs - self.model_function(self.training_inputs, parameters)
		derivative = np.transpose(self.training_inputs).dot(difference_vector)
		return derivative