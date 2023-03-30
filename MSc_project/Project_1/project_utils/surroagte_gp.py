from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

import numpy as np


class Surrogate_GP(object):

    def __init__(self, feature_names=None, mode='regression', classes=[], kernel_type='Matern'):

        self.feature_names = feature_names
        self.mode          = mode
        self.classes       = classes
        self.kernel_type   = kernel_type
        self.Num_Classes   = len(classes)

        
    def predict(self, X, y):
        
        if self.kernel_type == 'RBF':
            kernel = RBF()
        else:
            kernel = Matern()
        
        if self.mode == 'regression' or self.mode == 'Regression':
            
            GP = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 10, \
                                          optimizer='fmin_l_bfgs_b', normalize_y = False, copy_X_train=True)
            GP.fit(X, y)
    
            y_mean, y_std = GP.predict(X[0].reshape(1,-1), return_std=True)
        
            prediction    = np.empty(2)
            prediction[0] = y_mean
            prediction[1] = y_std
        
        else:
            
            GP = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer = 10, \
                                           optimizer='fmin_l_bfgs_b', normalize_y = False, copy_X_train=True)
            GP.fit(X, y)

            prediction = GP.predict(X[0].reshape(1,-1))
        

        print(prediction)
        return prediction    
    
    
    def predict_proba(self, X, y):
        
        if self.mode != 'classification' and self.mode != 'Classification':
            print('Classification only operation')
            return np.zeros(self.Num_Classes)
        
            
        if self.kernel_type == 'RBF':
            kernel = RBF()
        else:
            kernel = Matern()
        
        GP = GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer = 10, \
                                           optimizer='fmin_l_bfgs_b', normalize_y = False, copy_X_train=True)
        GP.fit(X, y)

        return GP.predict_proba(X[0].reshape(1,-1))
        
