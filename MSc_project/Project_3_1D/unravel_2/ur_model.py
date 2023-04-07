# Import Libraries
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

from sklearn.inspection import permutation_importance

from matplotlib import pyplot as plt

from unravel_2.acquisition_function import FUR_W

from project_utils.acq_data_capture import Acq_Data_1D

from copy import deepcopy



class UR_Model(object):

    def __init__(
        self,
        bbox_model,
        train_data,
        feature_names,
        categorical_features=[],
        mode="regression",
        sampling_optimize="Gaussian"
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

        self.feature_names = feature_names

        #self.categorical_features = categorical_features
        
        self.mode = mode
        
        self.gp_model = None
        
        self.std_x  = np.std (self.BB_train_data, axis=0)
        self.mean_x = np.mean(self.BB_train_data, axis=0)
        
        self.sampling_optimize = sampling_optimize
          
        self.acq_data = Acq_Data_1D()

        
    def BB_predict(self, X):
        return self.bbox_model.predict(np.array(X).ravel())
        
    
    def BB_predict_proba(self, X):
        return self.bbox_model.predict_proba(np.array(X).ravel())
    
    
    def Normalise_X(self, X):
        return ((X - self.mean_x) / self.std_x) 
    
    
    def Denormalise_X(self, X):
        return ((X * self.std_x) + self.mean_x) 
    
    def exp_predict_proba(self, X):
        if self.normalize:           
            return self.gp_model.predict_proba(X = np.array(self.Normalise_X(X)).reshape(1, -1))
        else:
            return self.gp_model.predict_proba(X = np.array(X).reshape(1, -1))
    
    def exp_predict(self, X):
        if self.normalize:
            return self.gp_model.predict(X = np.array(self.Normalise_X(X)).reshape(1, -1), return_std = True)
        else:
            return self.gp_model.predict(X = np.array(X).reshape(1, -1), return_std = True)
    
    
    def get_BB_model(self):
        return self.bbox_model
    
    def get_exp_model(self):
        model_copy = deepcopy(self.gp_model)
        return model_copy   
    
    def train_gaussian_process(self, x_train, y_train):

        if self.kernel_type == None or self.kernel_type == "RBF":
            kernel = RBF()
        elif self.kernel_type == "Matern":
            kernel = Matern()
    
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 10, optimizer='fmin_l_bfgs_b', normalize_y = False, copy_X_train=True)
        gpr.fit(x_train, y_train)
        
        return gpr
    
    
    def explain(self,
                X_init,
                kernel_type="RBF",
                max_iter=20,
                alpha="FUR_W",
                #jitter=5,
                normalize=False,
                #plot=False,
                interval=1,
                #verbosity=False,
                #maximize=False
               ):

        n_samples = 20
        

        self.kernel_type = kernel_type
        
        self.X_init   = X_init
        self.X_train  = X_init
        self.XN_train = self.Normalise_X(X_init)
        
        self.y_train = self.bbox_model.predict(self.X_train)
        self.y_train = np.array(self.y_train)
        
        self.normalize = normalize
        
        if self.normalize:
        
            self.gp_model = self.train_gaussian_process(self.XN_train, self.y_train)
            
            acq_function = FUR_W(X_init     = self.XN_train,
                                 std_x      = np.ones(len(self.feature_names)),
                                 n_samples  = n_samples,
                                 sample_opt = self.sampling_optimize,
                                 bounds      = interval)
        else:
            
            self.gp_model = self.train_gaussian_process(self.X_train, self.y_train)
            #print('X train ', self.X_train)

            
            acq_function = FUR_W(X_init     = self.X_init,
                                 std_x      = self.std_x, 
                                 n_samples  = n_samples,
                                 sample_opt = self.sampling_optimize,
                                 bounds      = interval)
        
        
        for iter in range(max_iter):
            
            x_next = acq_function.next_x(self.gp_model)
        
            y_next = self.gp_model.predict(x_next)
        
            fe_x0  = self.gp_model.predict(self.X_init)
            
            self.acq_data.new_X(X            = x_next,
                                y            = y_next,
                                fe_x0        = fe_x0,
                                acq_function = acq_function,
                                t1_t2        = True)

            

            if self.normalize:
                
                xd_next = self.Denormalise_X(x_next)

                self.X_train  = np.vstack([self.X_train,  xd_next])
                self.XN_train = np.vstack([self.XN_train, x_next])
            
                y_next = self.bbox_model.predict(xd_next)
            
                self.y_train = np.append(self.y_train, [y_next])
            
                self.gp_model = self.train_gaussian_process(self.XN_train, self.y_train)
            
            else:
                
                xn_next = self.Normalise_X(x_next)

                self.X_train  = np.vstack([self.X_train,  x_next])
                self.XN_train = np.vstack([self.XN_train, xn_next])
            
                y_next = self.bbox_model.predict(x_next)
            
                self.y_train = np.append(self.y_train, [y_next])
            
                self.gp_model = self.train_gaussian_process(self.X_train, self.y_train)
            
            
            
    def Plot(self, scores):
        plt.barh(y = self.feature_names, width = scores)
        plt.show()

            
    def permutation_importance(self, show_plot=False):
        
        results = permutation_importance(self.gp_model, self.X_train, self.y_train, scoring='neg_mean_squared_error')
        
        scores = results.importances_mean
        
        #print(scores)

        for i,v in enumerate(scores):
            print('%s:\t %.5f' % (self.feature_names[i],v))
            
        # plot feature importance
        if show_plot:
            self.Plot(scores)
            
        return scores             

    def get_acq_data(self):
        return self.acq_data
    

        
    def KL_imp(self, show_plot=False):
    
        return self.KLrel(X = self.X_train, model=self.gp_model, delta=1)
        
    def KLrel(self, X, model, delta):
        """Computes relevance estimates for each covariate using the KL method based on the data matrix X and a GPy model.
        The parameter delta defines the amount of perturbation used."""
        n = X.shape[0]
        p = X.shape[1]
        
        print(X.shape)
        return np.zeros([12])
        
        jitter = 1e-15
    
        # perturbation
        deltax = np.linspace(-delta,delta,3)
    
        # loop through the data points X
        relevances = np.zeros((n,p))
        
        for j in range(0, n):
    
            x_n = np.reshape(np.repeat(X[j,:],3),(p,3))
    
            # loop through covariates
            for dim in range(0, p):
                
                # perturb x_n
                x_n[dim,:] = x_n[dim,:] + deltax
                
                preddeltamean,preddeltavar = model.predict(x_n.T, return_std = True)
                mean_orig = np.asmatrix(np.repeat(preddeltamean[1],3)).T
                var_orig = np.asmatrix(np.repeat(preddeltavar[1],3)).T
                # compute the relevance estimate at x_n
                KLsqrt = np.sqrt(0.5*(var_orig/preddeltavar + np.multiply((preddeltamean.reshape(3,1)-mean_orig),(preddeltamean.reshape(3,1)-mean_orig))/preddeltavar - 1) + np.log(np.sqrt(preddeltavar/var_orig)) + jitter)
                relevances[j,dim] = 0.5*(KLsqrt[0] + KLsqrt[2])/delta
                
        # remove the perturbation
        x_n[dim,:] = x_n[dim,:] - deltax
            
        return relevances
    
    
    