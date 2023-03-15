# Import Libraries
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

from sklearn.inspection import permutation_importance

from matplotlib import pyplot as plt

from unravel_2.acquisition_function import FUR_W

class UR_Model(object):

    def __init__(
        self,
        bbox_model,
        train_data,
        feature_names,
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

        self.feature_names = feature_names

        #self.categorical_features = categorical_features
        
        self.mode = mode
        
        self.gp_model = None
        
        self.std_x = np.std(self.BB_train_data, axis=0)
               

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
    
    def train_gaussian_process(self, x_train, y_train):

        if self.kernel_type == None or self.kernel_type == "RBF":
            kernel = RBF()
        elif self.kernel_type == "Matern":
            kernel = Matern()
    
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 5, normalize_y = False)
        gpr.fit(x_train, y_train)
        
        return gpr
    
    
    def explain(
        self,
        X_init,
        kernel_type="RBF",
        max_iter=20,
        alpha="FUR_W",
        #jitter=5,
        normalize=True,
        #plot=False,
        interval=1,
        #verbosity=False,
        #maximize=False
    ):

        n_samples = 20
        

        self.kernel_type = kernel_type
        
        self.X_train = X_init
        
        self.y_train = self.bbox_model.predict(self.X_train)
        self.y_train = np.array(self.y_train)
        
        self.gp_model = self.train_gaussian_process(self.X_train, self.y_train)
        
        acq_function = FUR_W(X_init = X_init, std_x = self.std_x)
        
        for iter in range(max_iter):
            
            x_next = acq_function.next_x(self.gp_model, n_samples=n_samples)
            
          
            self.X_train = np.vstack([self.X_train, x_next])
            
            y_next = self.bbox_model.predict(x_next)
            
            self.y_train = np.append(self.y_train, [y_next])
            
            self.gp_model = self.train_gaussian_process(self.X_train, self.y_train)
            
            
            
    def Plot(self, scores):
        plt.barh(y = self.feature_names, width = scores)
        plt.show()

            
    def permutation_importance(self, show_plot=False):
        
        results = permutation_importance(self.gp_model, self.X_train, self.y_train, scoring='neg_mean_squared_error')
        
        scores = results.importances_mean
        
        print(scores)

        for i,v in enumerate(scores):
            print('%s:\t %.5f' % (self.feature_names[i],v))
            
        # plot feature importance
        if show_plot:
            self.Plot(scores)
            
        return scores
    
    
    def KL_importance(self, delta=1.0, show_plot=False):
        # Code credit: https://github.com/topipa/gp-varsel-kl-var
        
        # Storing the surrogate dataset generated through the BO routine
        #X_surrogate = f_optim.get_evaluations()[0]
        # Use X_Train
        
        # y_surrogate = f_e.get_evaluations()[1]

        # Storing the GP model trained during the BO routine
        #gp_model = f_optim.model
        # Use self.gp_model

        n_samples = self.X_train.shape[0]
        n_features = self.X_train.shape[1]

        jitter = 1e-15

        # perturbation
        deltax = np.linspace(-delta, delta, 3)

        # loop through the data points X
        relevances = np.zeros((n_samples, n_features))

        for sample in range(0, n_samples):

            x_n = np.reshape(np.repeat(self.X_train[sample, :], 3), (n_features, 3))
            # loop through covariates
            for dim in range(0, n_features):

                # perturb x_n
                x_n[dim, :] = x_n[dim, :] + deltax

                preddeltamean, preddeltavar = self.gp_model.predict(X = x_n.T, return_std = True)
                
                mean_orig = np.asmatrix(np.repeat(preddeltamean[1], 3)).T
                var_orig = np.asmatrix(np.repeat(preddeltavar[1] ** 2, 3)).T
                # compute the relevance estimate at x_n
                KLsqrt = np.sqrt(
                    0.5
                    * (
                        var_orig / preddeltavar
                        + np.multiply(
                            (preddeltamean.reshape(3, 1) - mean_orig),
                            (preddeltamean.reshape(3, 1) - mean_orig),
                        )
                        / preddeltavar
                        - 1
                    )
                    + np.log(np.sqrt(preddeltavar / var_orig))
                    + jitter
                )
                relevances[sample, dim] = 0.5 * (KLsqrt[0] + KLsqrt[2]) / delta

                # remove the perturbation
                x_n[dim, :] = x_n[dim, :] - deltax

            scores = np.mean(relevances, axis=0)

        return scores
            
        # plot feature importance
        if show_plot:
            self.Plot(scores)
            
        return scores