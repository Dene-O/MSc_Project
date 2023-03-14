# Import Libraries
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

from sklearn.inspection import permutation_importance

from matplotlib import pyplot

from unravel_2.acquisition_function import FUR_W

class UR_Model(object):

    def __init__(
        self,
        bbox_model,
        train_data,
        categorical_features=[],
        mode="regression",
    ):
        """
        Basic constructor

        Args:
            bbox_model (Any Scikit-Learn model): Black box prediction model
            (BB_)train_data (Numpy Array): Data used to train bbox_model
            categorical_features (List, optional): List containing indices of categorical features
            mode (str, optional): Explanation mode. Can be 'regression', or 'classification'
        """

        self.bbox_model = bbox_model
        self.BB_train_data = train_data
        
        #self.categorical_features = categorical_features
        
        self.mode = mode
        
        #self.X_Train = []
        
        self.gp_model = None
        
        self.std_x = np.std(self.BB_train_data, axis=0)
        print('STD: ', self.std_x)
                

    def BB_predict(self, X):
        return self.bbox_model.predict(np.array(X).ravel())
        
    
    def BB_predict_proba(self, X):
        return self.bbox_model.predict_proba(np.array(X).ravel())
    
    
    def exp_predict(self, X):
        return self.gp_model.predict(X = np.array(X).reshape(1, -1), return_std = True)
        
    
    def exp_predict_proba(self, X):
        return self.gp_model.predict_proba(X = np.array(X).reshape(1, -1), return_std = True)
    
    
    def get_BB_model(self):
        return self.bbox_model
    
    def get_exp_model(self):
        return self.gp_model
    
    def get_debug_data(self):
        return self.debug_data
    
    
    def train_gaussian_process(self, x_train, y_train):

        if self.kernel_type == None or self.kernel_type == "RBF":
            kernel = RBF()
        elif self.kernel_type == "Matern":
            kernel = Matern()
    
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 5, normalize_y = False)
        gpr.fit(x_train,y_train)
        
        return gpr
    
    
    def explain(
        self,
        X_init,
        feature_names,
        kernel_type="RBF",
        max_iter=20,
        alpha="FUR_W",
        #jitter=5,
        normalize=True,
        #plot=False,
        interval=1,
        #verbosity=False,
        #maximize=False,
        importance_method="ARD",
        delta=1,
    ):

        n_samples = 20
        
        self.debug_data = np.empty([max_iter, n_samples+1, 4])


        self.kernel_type = kernel_type
        
        X_train = X_init
        
        y_train = self.bbox_model.predict(X_train)
        y_train = np.array(y_train)
        
        self.gp_model = self.train_gaussian_process(X_train, y_train)
        
        acq_function = FUR_W(X_init = X_init, std_x = self.std_x)
        
        for iter in range(max_iter):
            
            x_next, debug_data = acq_function.next_x(self.gp_model, n_samples=n_samples)
            
            self.debug_data[iter,:,:] = debug_data
            
            X_train = np.vstack([X_train, x_next])
            
            y_next = self.bbox_model.predict(x_next)
            
            y_train = np.append(y_train, [y_next])
            
            self.gp_model = self.train_gaussian_process(X_train, y_train)
        
        
        results = permutation_importance(self.gp_model, X_train, y_train, scoring='neg_mean_squared_error')
        
        importance = results.importances_mean

        for i,v in enumerate(importance):
            print('Feature: %0d, Score: %.5f' % (i,v))
            
        # plot feature importance
        #pyplot.bar([x for x in range(len(importance))], importance)
        #pyplot.show()