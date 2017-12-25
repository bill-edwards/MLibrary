class GradientDescent(object):

	def __init__(self, options):
		self.step_size = options['step_size']
		self.number_of_steps = options['number_of_steps']

	def optimise(self, cost, initial_parameters):
		parameters = initial_parameters
		parameter_history = [initial_parameters]
		cost_history = [cost.cost_function(initial_parameters)]
		for step in range(self.number_of_steps):
			parameters = parameters + (self.step_size * cost.first_derivative(parameters))
			parameter_history.append(parameters)
			cost_this_step = cost.cost_function(parameters)
			cost_history.append(cost_this_step)
			print 'step', step, ':', cost_this_step
		return (parameter_history, cost_history)