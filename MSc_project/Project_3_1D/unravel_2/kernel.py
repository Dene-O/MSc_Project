import sklearn
import numpy as np

def get_kernel_sklearn(kernel_name, n_inputs): 

    hp_low = 1e-10
    hp_high = 1e9

    hp_bounds = np.zeros(shape = (n_inputs,2))
    hp_bounds[:,0] = hp_low
    hp_bounds[:,1] = hp_high

    len_scale = np.ones(n_inputs)

    #default
    kernel = sklearn.gaussian_process.kernels.RBF(length_scale=len_scale, length_scale_bounds=hp_bounds)
    
    if(kernel_name == 'RBF'):
        kernel = sklearn.gaussian_process.kernels.RBF(length_scale=len_scale, length_scale_bounds=hp_bounds)
        
    elif(kernel_name == 'RQ'):
        kernel = sklearn.gaussian_process.kernels.RationalQuadratic(length_scale=len_scale, length_scale_bounds=hp_bounds, alpha_bounds=hp_bounds)
        
    elif(kernel_name == 'Matern_1'):
        kernel = sklearn.gaussian_process.kernels.Matern(length_scale=len_scale, length_scale_bounds=hp_bounds, nu=0.5)
        
    elif(kernel_name == 'Matern_3'):
        kernel = sklearn.gaussian_process.kernels.Matern(length_scale=len_scale, length_scale_bounds=hp_bounds, nu=1.5)
        
    elif(kernel_name == 'Matern_5'):
        kernel = sklearn.gaussian_process.kernels.Matern(length_scale=len_scale, length_scale_bounds=hp_bounds, nu=2.5)
        
    elif(kernel_name == 'Periodic'):
        kernel = sklearn.gaussian_process.kernels.ExpSineSquared(length_scale=len_scale, length_scale_bounds=hp_bounds)
        
        
    return kernel
