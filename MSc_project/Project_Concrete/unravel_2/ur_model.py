# Import Libraries
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF

from sklearn.inspection import permutation_importance

from sklearn.linear_model import LinearRegression

from matplotlib import pyplot as plt

from unravel_2.acquisition_function import FUR_W

from unravel_2.kernel import get_kernel_sklearn

from project_utils.acq_data_capture import Acq_Data_1D_For
from project_utils.acq_data_capture import Acq_Data_1D
from project_utils.acq_data_capture import Acq_Data_2D
from project_utils.acq_data_capture import Acq_Data_2D_For
from project_utils.acq_data_capture import Acq_Data_nD

from project_utils.GP_varsel import KLrel
from project_utils.GP_varsel import VARrel

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
        self.N_features    = len(feature_names)

        #self.categorical_features = categorical_features
        
        self.mode = mode
        
        self.gp_model = None
        
        self.std_x  = np.std (self.BB_train_data, axis=0)
        self.mean_x = np.mean(self.BB_train_data, axis=0)
        
        self.sampling_optimize = sampling_optimize
          
        # 
        self.acq_data = None

        
    def BB_predict(self, X):
        return self.bbox_model.predict(np.array(X).ravel())
        
    
    def BB_predict_proba(self, X):
        return self.bbox_model.predict_proba(np.array(X).ravel())
    
    
    def exp_predict_proba(self, X):
        return self.gp_model.predict_proba(X = np.array(X).reshape(1, -1))
    
    def exp_predict(self, X):
        return self.gp_model.predict(X = np.array(X).reshape(1, -1), return_std = True)
    
    
    def get_BB_model(self):
        return self.bbox_model
    
    def get_exp_model(self):
        model_copy = self.train_gaussian_process(self.X_train, self.y_train)
        return model_copy   
    
    def get_exp_L(self):
        L_copy = deepcopy(self.gp_model.L_)
        return L_copy   
    
    def train_gaussian_process(self, x_train, y_train, n_features=0):

        
        if self.kernel_type == None or self.kernel_type == "RBF":
            kernel = RBF()
        elif self.kernel_type == "Matern":
            kernel = Matern()
        else:
            if n_features == 0:
                n_features = self.N_features
                
            kernel = get_kernel_sklearn(self.kernel_type, n_features)
        
    
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer = 10, optimizer='fmin_l_bfgs_b', normalize_y = False, copy_X_train=True)
        gpr.fit(x_train, y_train)
        
        return gpr
    
    
    
    def explain(self,
                X_init,
                Dimension='Multi D',
                kernel_type="Matern_3",
                max_iter=20,
                alpha="FUR_W",
                #jitter=5,
                #plot=False,
                interval=1,
                weight=None,
                #verbosity=False,
                #maximize=False
                ):

        n_samples = 20
        

        self.kernel_type = kernel_type
        
        self.X_init   = X_init
        self.X_train  = X_init
        
        self.y_train = self.bbox_model.predict(self.X_train)
        self.y_train = np.array(self.y_train)
               
        bounds = np.empty([2,self.N_features])
        
        bounds[0,:] = self.X_init - self.std_x
        bounds[1,:] = self.X_init + self.std_x
        #print('bounds',bounds)
        
        if Dimension == 'One':
            self.acq_data = Acq_Data_1D(X_Init = X_init, bounds = bounds, BB_Model = self.bbox_model)
        elif Dimension == 'One_For':
            self.acq_data = Acq_Data_1D_For()
        elif Dimension == 'Two':
            self.acq_data = Acq_Data_2D(X_Init = X_init, bounds = bounds, BB_Model = self.bbox_model)
        elif Dimension == 'Two_For':
            self.acq_data = Acq_Data_2D_For()
        else:
            self.acq_data = Acq_Data_nD(X_Init = X_init, bounds = bounds, BB_Model = self.bbox_model)


        self.gp_model = self.train_gaussian_process(self.X_train, self.y_train)
        #print('X train ', self.X_train)

            
        acq_function = FUR_W(X_init     = self.X_init,
                             mean_x     = self.mean_x, 
                             std_x      = self.std_x, 
                             weight     = weight,
                             n_samples  = n_samples,
                             sample_opt = self.sampling_optimize,
                             bounds     = interval)
        
        
        for iter in range(max_iter):
            
            x_next = acq_function.next_x(self.gp_model)
        
            y_next = self.gp_model.predict(x_next)
        
            fe_x0  = self.gp_model.predict(self.X_init)
            
            self.acq_data.new_X(X            = x_next,
                                y            = y_next,
                                fe_x0        = fe_x0,
                                acq_function = acq_function,
                                t1_t2        = True)

            
            self.X_train  = np.vstack([self.X_train,  x_next])
            
            y_next = self.bbox_model.predict(x_next)
            
            self.y_train = np.append(self.y_train, [y_next])
            
            self.gp_model = self.train_gaussian_process(self.X_train, self.y_train)
            
            
            
    def Plot(self, scores, filename=""):
        plt.barh(y = self.feature_names, width = scores)

        if filename != "":
            plt.savefig(fname=filename)
        
        plt.show()

            
    def permutation_importance(self, show_plot=False):
        
        results = permutation_importance(self.gp_model, self.X_train, self.y_train, scoring='neg_mean_squared_error')
        
        self.perm_scores = results.importances_mean
        

#        for i,v in enumerate(scores):
#            print('%s:\t %.5f' % (self.feature_names[i],v))
            
        # plot feature importance
        if show_plot:
            self.Plot(self.perm_scores)
            
        return self.perm_scores             

    def get_acq_data(self):
        return self.acq_data
    

        
    def KL_imp(self, show_plot=False):
        
        self.KL_scores =  KLrel(X = self.X_train, model=self.gp_model, delta=1)
        
        return self.KL_scores
        

        
    def Var_imp(self, show_plot=False):
        
        self.Var_scores = VARrel(X=self.X_train, model=self.gp_model, nquadr=12, pointwise = False)
        
        return self.Var_scores
    
        

        
    def KLrel_a(self, X, model, delta):
        """Computes relevance estimates for each covariate using the KL method based on the data matrix X and a GPy model.
        The parameter delta defines the amount of perturbation used."""
        n = X.shape[0]
        p = X.shape[1]
        
        #print(X.shape)
        
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

                preddeltamean = preddeltamean.reshape(3,1)
                preddeltavar  = preddeltavar.reshape(3,1)
                
                mean_orig = np.asmatrix(np.repeat(preddeltamean[1],3)).T
                var_orig = np.asmatrix(np.repeat(preddeltavar[1],3)).T
                    
                # compute the relevance estimate at x_n
                KLsqrt = np.sqrt(0.5*(var_orig/preddeltavar + np.multiply((preddeltamean.reshape(3,1)-mean_orig),(preddeltamean.reshape(3,1)-mean_orig))/preddeltavar - 1) + np.log(np.sqrt(preddeltavar/var_orig)) + jitter)
                
                relevances[j,dim] = 0.5*(KLsqrt[0] + KLsqrt[2])/delta
                
        # remove the perturbation
        x_n[dim,:] = x_n[dim,:] - deltax

        self.KL_scores = np.mean(relevances, axis=0)

        return self.KL_scores
    
    

    def del_1_rel(self):
        
        variance = np.empty(self.N_features)
        
        y = self.gp_model.predict(self.X_init)
        
        for feature in range(self.N_features):
            
            X_p = deepcopy(self.X_init.ravel())

            X_p[feature] = 0
            
            X_p = X_p.reshape([1,self.N_features])
                       
            y_p = self.gp_model.predict(X_p)
           
            variance[feature] = np.abs(y - y_p)
            
        
        self.del_1_variance = variance
        
        self.del_1_scores = variance / np.mean(variance)
        
        return self.del_1_scores
            

    def get_del_1_variance(self):
        return self.del_1_variance
        
            
            
    def del_2_rel(self):
        
        variance = np.empty(self.N_features)
        
        y = self.gp_model.predict(self.X_init)
        
        N_Samples = self.X_train.shape[0]
        
        for feature_outer in range(self.N_features):

            X_train_p = np.empty([N_Samples, self.N_features - 1])
            
            for feature_inner in range(feature_outer):
                
                X_train_p[:,feature_inner] = self.X_train[:,feature_inner]

            for feature_inner in range(feature_outer + 1, self.N_features):
                
                X_train_p[:,feature_inner - 1] = self.X_train[:,feature_inner]
                
                
            GP = self.train_gaussian_process(X_train_p, self.y_train, self.N_features - 1)

            X_p = X_train_p[0,:].reshape(1,-1)
                       
            y_p = GP.predict(X_p)
           
            variance[feature_outer] = np.square(y - y_p)
            
        
        self.del_2_scores = variance / np.mean(variance)
        
        return self.del_2_scores
            
            
    def Lin_scores(self):
        
#        print('Shapes: ', self.X_init.shape, self.std_x.shape)
        X_inits = np.repeat(self.X_init, self.X_train.shape[0] ,axis = 0)
        SDs     = np.repeat(self.std_x.reshape(1,-1),  self.X_train.shape[0] ,axis = 0)
        
        weights = 1 + np.square((self.X_train - X_inits) / SDs)
        
#        print('WS: ', weights.shape)
#        print('W:  ', weights)   
        
        min_value = np.min(weights)
        
        weights = np.mean((min_value / weights), axis = 1)
#        print('WS: ', weights.shape)
#        print('W:  ', weights)   
        
        LR = LinearRegression()
              
        LR.fit(self.X_train, self.y_train, weights)
        
        self.lin_scores = LR.coef_
        
        return self.lin_scores

    
    def plot_scores(self, title, filename=""):
        
        fig, ax = plt.subplots()
        
        title = title
        
        ax.set_title(title)
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Score')
        
        ax.set_ylim(-0.05,1.05)
        
        x = np.arange(self.N_features)

        perm = self.perm_scores / np.max(self.perm_scores)
        ax.plot(x, perm, label='Perm')           
        
        KL = self.KL_scores / np.max(self.KL_scores)
        ax.plot(x, KL, label='KL')           
        
        Var = self.Var_scores / np.max(self.Var_scores)
        ax.plot(x, Var, label='Var')           
        
        Del1 = self.del_1_scores / np.max(self.del_1_scores)
        ax.plot(x, Del1, label='Del1')           
        
        Del2 = self.del_2_scores / np.max(self.del_2_scores)
        ax.plot(x, Del2, label='Del2')           
        
        Lin = abs(self.lin_scores / np.max(abs(self.lin_scores)))
        ax.plot(x, Lin, label='Lin')           
        
        ax.legend()

        fig.tight_layout()

        if filename != "":
            plt.savefig(fname=filename)
        
        plt.show() 
        
        
    def Y_Consistancy(self, N_points, std_bound):       
        
        mid_index   = int(N_points / 2)
        index_range = 2 * mid_index + 1
#        print('N_points: ', index_range)
                
        y_p = np.empty([index_range,2])
        
        for idx in range(index_range):
            
            x_perturbed = std_bound * (idx - mid_index) / mid_index
            
#            print(x_perturbed)
            x_perturbed = self.X_init + x_perturbed * self.std_x
                
#            print('X0   ',self.X_init)
#            print('XDIFF',x_perturbed)
                
#            y_p[idx,:] = self.gp_model.predict(x_perturbed.reshape(1, -1), return_std = True)
            yy1, yy2 = self.gp_model.predict(x_perturbed.reshape(1, -1), return_std = True)
#            print('Yp: ',y_p[idx,:])

#        print('YP: ',y_p)
        return y_p
