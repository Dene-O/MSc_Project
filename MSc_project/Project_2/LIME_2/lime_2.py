# Import Libraries
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

from scipy.stats.qmc import LatinHypercube

from matplotlib import pyplot as plt

from copy import deepcopy


class LIME_Model(object):
    
    prime_sq_list = [25, 49, 121, 169, 289, 361, 529, 841, 961, 1369, 1681, 1849, 2209, 2809, 3481, 3721, 4489, 5041]


    def __init__(
        self,
        bbox_model,
        train_data,
        feature_names,
        categorical_features= [],
        class_names         = [],
        mode                = "regression",
        sampling            = "Gaussian"
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
        
        self.class_names = class_names
        
        self.mode = mode
        
        self.LM_model = None
        
        self.std_x  = np.std (self.BB_train_data, axis=0)
        self.mean_x = np.mean(self.BB_train_data, axis=0)
        
        self.sampling = sampling


    def explain(self,
                X_init,
                #plot=False,
                sample_around_instance=True,
                #verbosity=False,
                #maximize=False,
                bounds=1,
                LHC_strength=1,
                Student_T_DF=1,
                N_samples = 2000,
                ):

        self.X_init = X_init     

        if self.sampling == "Gaussian" or self.sampling == "gaussian":
            
            self.sampling = "Gaussian"
                        
            sampled_dist = np.random.randn(N_samples, self.N_features) * bounds
            
            self.N_samples = N_samples

        elif self.sampling == "Student_T" or self.sampling == "Standard_T" or self.sampling == "std_t":
            
            self.sampling = "Student_T"
                        
            sampled_dist = np.random.standard_t(df = Student_T_DF, size = [N_samples, self.N_features]) * bounds

            self.N_samples = N_samples

        elif self.sampling == "LatinHyperCube" or self.sampling == "latinhypercube" or self.sampling == "LHC":
                        
            self.sampling = "Latin HC"
                        
            if LHC_strength == 1:
                self.N_samples = N_samples

            else:
                self.N_samples = LIME_Model.prime_sq_list[-1]
            
                for prime_sq in Acq_Base.prime_sq_list:
                    if prime_sq > N_samples:
                        self.N_samples = prime_sq
                        break
                        
                    
            sampler = LatinHypercube(d = self.N_features, strength = LHC_strength)
            
            sampled_dist = sampler.random(n = self.N_samples) * 2 * bounds - bounds
            
        else:
            print('UKNOWN SAMPLING DISTRIBUTION')
            return
            
    
        weights   = np.square(sampled_dist) + 1

        weights   = np.mean(weights, axis = 1)
        
        min_value = np.min(weights)
        
        self.weights   = min_value / weights
        
#        print('WS: ', weights.shape)
#        print('W:  ', weights)   
#        print('mean:  ', np.mean(weights))   
#        print('min:   ', np.min(weights))   
#        print('max:   ', np.max(weights))   
        

        if sample_around_instance:
                                
            self.X_train = sampled_dist * self.std_x + X_init
                        
        else:
            
            self.X_train = sampled_dist * self.std_x + self.mean_x


        self.y_train = self.bbox_model.predict(self.X_train)
            
        self.LM_model = LinearRegression()
              
        self.LM_model.fit(self.X_train, self.y_train, self.weights)
        
        self.lime_scores = self.LM_model.coef_
    
    
    def permutation_importance(self, show_plot=False):
        
        results = permutation_importance(self.LM_model, self.X_train, self.y_train, scoring='neg_mean_squared_error')
        
        self.perm_scores = results.importances_mean
        

#        for i,v in enumerate(scores):
#            print('%s:\t %.5f' % (self.feature_names[i],v))
            
        # plot feature importance
        if show_plot:
            self.Plot(self.perm_scores)
            
        return self.perm_scores             

        
    def del_1_rel(self):
        
        variance = np.empty(self.N_features)
        
        y = self.LM_model.predict(self.X_init)
        
        for feature in range(self.N_features):
            
            X_p = deepcopy(self.X_init.ravel())

            X_p[feature] = 0
            
            X_p = X_p.reshape([1,self.N_features])
                       
            y_p = self.LM_model.predict(X_p)
           
            variance[feature] = np.square(y - y_p)
            
        
        self.del_1_scores = variance / np.mean(variance)
        
        return self.del_1_scores
            
            
            
    def del_2_rel(self):
        
        variance = np.empty(self.N_features)
        
        y = self.LM_model.predict(self.X_init)
        

        X_Init_p = np.empty([1,self.N_features - 1])
        
        for feature_outer in range(self.N_features):

            X_train_p = np.empty([self.N_samples, self.N_features - 1])            
            
            for feature_inner in range(feature_outer):
                
                X_train_p[:,feature_inner] = self.X_train[:,feature_inner]
                
                X_Init_p[0,feature_inner]  = self.X_init[0,feature_inner]

            for feature_inner in range(feature_outer + 1, self.N_features):
                              
                X_train_p[:,feature_inner - 1] = self.X_train[:,feature_inner]
                
                X_Init_p[0,feature_inner - 1]  = self.X_init[0,feature_inner]


                
            LM_model = LinearRegression()
              
            LM_model.fit(X_train_p, self.y_train, self.weights)
        
            X_p = X_train_p[0,:].reshape(1,-1)
                       
            y_p = LM_model.predict(X_Init_p)
           
            variance[feature_outer] = np.square(y - y_p)
            
        
        self.del_2_scores = variance / np.mean(variance)
        
        return self.del_2_scores
            
            
         
        
    def get_LIME_scores(self):
        return self.lime_scores
          
        
    def BB_predict(self, X):
        return self.bbox_model.predict(np.array(X).ravel())
        
    
    def BB_predict_proba(self, X):
        return self.bbox_model.predict_proba(np.array(X).ravel())
    
    
    def exp_predict_proba(self, X):
        return self.LM_model.predict_proba(X = np.array(X).reshape(1, -1))
    
    def exp_predict(self, X):
        return self.LM_model.predict(X = np.array(X).reshape(1, -1))
    
        
    def get_BB_model(self):
        return self.bbox_model
    
    def get_exp_model(self):
        model_copy = deepcopy(self.LM_model)
        return model_copy   


    def plot_scores(self, title):
    
        title = title + " SMP: " + self.sampling + ", " + str(self.N_samples) + " Samples"
        fig, ax = plt.subplots()
    
        ax.set_title(title)
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Score')
        
        ax.set_ylim(-0.05,1.05)
        
        x = np.arange(self.N_features)

        perm = self.perm_scores / np.max(self.perm_scores)
        ax.plot(x, perm, label='Perm')           
        
        Del1 = self.del_1_scores / np.max(self.del_1_scores)
        ax.plot(x, Del1, label='Del1')           
        
        Del2 = self.del_2_scores / np.max(self.del_2_scores)
        ax.plot(x, Del2, label='Del2')           
        
        lime = abs(self.lime_scores / np.max(abs(self.lime_scores)))
        ax.plot(x, lime, label='LIME')           
        
        ax.legend()
        
        fig.tight_layout()

        plt.show()
        
        
