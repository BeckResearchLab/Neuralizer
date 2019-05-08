import data_process as dp
import param_record as pr

def param_store(proj_name,X_var,Y_var):
	param_list = {'X_var':X_var,'Y_var':Y_var}
	pr.check_write(param_list,"%s_param.json"%(proj_name))

	
