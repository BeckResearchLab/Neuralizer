import model_create as mc

def test_make_combo():
	parameter_combo = mc.make_combo()
	assert len(parameter_combo) == 9; "The combination of hyperparameter choices is not correct"

def test_make_pairwise_list():
	max_depth =2
	options = [1,5,10]
	combinations = mc.make_pairwise_list(max_depth=max_depth, options=options)
	length = len(options)**max_depth
	
	assert len(combinations) == length,'not every situation is considered'

	assert combinations[0][1] == 1, 'The hyperparameter is not as chosen'
	assert combinations[5][0] == 5, 'The hyperparameters are not combined as designed'
	assert combinations[2][1] == 10, 'The hyperparameters are not combined as designed'