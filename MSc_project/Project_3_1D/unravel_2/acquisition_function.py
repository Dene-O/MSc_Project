# Import Libraries
import numpy as np

import matplotlib.pyplot as plt

from scipy.stats.qmc import LatinHypercube

class Acq_Base(object):

    prime_sq_list = [9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369]

    def __init__(self, X_init):
        pass
    
    def next_x(self):
        pass
    
    
    
class FUR_W(Acq_Base):
    
   
    def __init__(self, X_init, std_x, weight=None, distribution="Gaussian", n_samples=200, strength=1):
        
        self.X_init = X_init
        
        self.iter = 1
        
        self.n_features = X_init.shape[1]
        
        self.std_x = std_x
        
        if weight == None:
            self.weight = [0.5, 0.5]
        
        elif isinstance(weight, float):
            self.weight = [weight, 1.0 - weight]
            
        else:
            self.weight = weight

        if distribution == "Gaussian":
            
            self.n_samples    = n_samples
            self.sampled_dist = np.random.randn(n_samples, self.n_features)

        elif distribution == "LatinHyperCube":
            
            if strength == 1:
                self.n_samples = n_samples
            
            else:
                self.n_samples = Acq_Base.prime_sq_list[-1]
            
                for prime_sq in Acq_Base.prime_sq_list:
                    if prime_sq > n_samples:
                        self.n_samples = prime_sq
                        break
                    
            sampler = LatinHypercube(d = self.n_features, strength = strength)
            
            self.sampled_dist = sampler.random(n = self.n_samples)
    

    def acq_function(self, x):
        
        mean_p, std_p = self.gp_model.predict(X = x, return_std = True)
        
        self.delta = np.random.randn()#self.n_features)##########################################################################
        
        f_acqu = self.weight[0] * -np.linalg.norm(x - self.X_init - self.std_x * self.delta / np.log(self.iter)) + \
                 self.weight[1] * std_p
                                         
        return f_acqu
    

    def next_x(self, gp_model):
        self.iter = self.iter + 1
        
        self.gp_model = gp_model

        self.X_diffs = np.empty([self.n_samples+1])
        self.acqu_fs = np.empty([self.n_samples+1])
        
        max_acq_value = self.acq_function(self.X_init)
        x_next        = self.X_init
        
        self.X_diffs[0] = 0
        self.acqu_fs[0] = max_acq_value
        
        
        for sample in range(self.n_samples):
            
            x = self.X_init + self.std_x * self.sampled_dist[sample,:]
        
            self.X_diffs[sample+1] = self.Calculate_X0_Distance(x)

            acq_value = self.acq_function(x)

            self.acqu_fs[sample+1] = acq_value
            
            if acq_value > max_acq_value:
                max_acq_value = acq_value
                x_next = x
                #print(x_next)

                
        self.Plot_X_Diff_Acq()
        
        return x_next
                
    
    def Set_Weight(self, weight):
        
        if isinstance(weight, float):
            self.weight = [weight, 1.0 - weight]
            
        else:
            self.weight = weight            


    def Calculate_X0_Distance(self, X):
        
        distance_sq = np.square((self.X_init - X)/self.std_x)
    
        distance_RMS = np.sqrt(np.mean(distance_sq))
        
        return distance_RMS
        
        
    def Plot_X_Diff_Acq(self):
        
        plt.scatter(self.X_diffs, self.acqu_fs)
    
    
    