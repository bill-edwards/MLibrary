import numpy as np

def generate_linear_test_data(parameters, num_datapoints, std_dev=0.1):
	parameters = np.array(parameters).reshape(len(parameters), 1) # Applying np.array to a list returns a rank-1 objects - we need a rank-2 matrix.
	num_features = len(parameters)
	features = np.random.randn(num_datapoints, num_features)
	expected_outputs = np.dot(features, parameters)
	actual_outputs = expected_outputs + np.random.normal(0, std_dev, expected_outputs.shape)
	return (features, actual_outputs)