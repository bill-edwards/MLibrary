import numpy as np

# Functions for generating data for testing models.

# Helper function to randomly generate datapoints, with the expected output calculated precisely using the linear model.
def generate_linear_data(parameters, num_datapoints):
	parameters = np.array(parameters).reshape(len(parameters), 1) # Applying np.array to a list returns a rank-1 objects - we need a rank-2 matrix.
	num_features = len(parameters)
	inputs = np.random.randn(num_datapoints, num_features)
	inputs[:,0] = 1
	expected_outputs = np.dot(inputs, parameters)
	return (inputs, expected_outputs)

# Generates test data for a regression problem, by adding random error to calculated expected ouptuts.
def generate_continuous_datapoints(parameters, num_datapoints, std_dev=0.1):
	inputs, expected_outputs = generate_linear_data(parameters, num_datapoints)
	actual_outputs = expected_outputs + np.random.normal(0, std_dev, expected_outputs.shape)
	return (inputs, actual_outputs)

# Generates test data for a binary classification problem, assigning 0 or 1 to each datapoint according to probabilities calculated from its expected output.
def generate_discrete_datapoints(parameters, num_datapoints):
	inputs, expected_outputs = generate_linear_data(parameters, num_datapoints)
	output_probabilities = sigmoid(expected_outputs)
	random_vector = np.random.uniform(size=(num_datapoints, 1)) # Generate random numbers from a uniform distribution.
	actual_outputs = map(lambda x: 1 if x else 0, random_vector < output_probabilities)
	return (inputs, actual_outputs)

# Miscellaneous functions used by other classes.

def sigmoid(z):
	return 1/(1 + np.exp(-z))