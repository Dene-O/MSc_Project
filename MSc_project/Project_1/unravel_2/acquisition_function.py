# Import Libraries
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.qmc import LatinHypercube
from scipy.optimize  import minimize
from scipy.optimize  import Bounds

class Acq_Base(object):

    prime_sq_list = [9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369]

    def __init__(self, X_init):
        pass
    
    def next_x(self):
        pass
    
    
    
class FUR_W(Acq_Base):
    
   
    def __init__(self, X_init, std_x, weight=None, sample_opt="Gaussian", n_samples=200, LHC_strength=1, bounds=1):
        
        self.X_init = X_init
        
        self.iter  = 1
        self.delta = np.random.randn()
        
        self.n_features = X_init.shape[1]
        
        self.std_x = std_x
        
        self.X_next = X_init
        
        if weight == None:
            self.weight = [0.5, 0.5]
        
        elif isinstance(weight, float):
            self.weight = [weight, 1.0 - weight]
            
        else:
            self.weight = weight

        if sample_opt == "Gaussian" or sample_opt == "gaussian":
            
            if bounds == None: bounds = 1
            
            self.n_samples    = n_samples
            self.sampled_dist = np.random.randn(n_samples, self.n_features) * bounds
            self.sampling     = True

        elif sample_opt == "Uniform" or sample_opt == "uniform":
            
            if bounds == None: bounds = 1
            
            self.n_samples    = n_samples
            self.sampled_dist = np.random.random([n_samples, self.n_features]) * 2 * bounds - bounds
            self.sampling     = True

        elif sample_opt == "LatinHyperCube" or sample_opt == "latinhypercube" or sample_opt == "LHC":
            
            if bounds == None: bounds = 1
            
            self.sampling = True

            if LHC_strength == 1:
                self.n_samples = n_samples
            
            else:
                self.n_samples = Acq_Base.prime_sq_list[-1]
            
                for prime_sq in Acq_Base.prime_sq_list:
                    if prime_sq > n_samples:
                        self.n_samples = prime_sq
                        break
                    
            sampler = LatinHypercube(d = self.n_features, strength = LHC_strength)
            
            self.sampled_dist = sampler.random(n = self.n_samples) * 2 * bounds - bounds
    
        elif sample_opt == "Optimize" or sample_opt == "optimize" or sample_opt ==  "opt" or sample_opt == "L-BFGS-B":
            
            self.sampling   = False
            self.opt_method = "L-BFGS-B"
            
            if bounds == None:
                self.bounds = None
            else:
                lower = X_init - std_x * bounds
                upper = X_init + std_x * bounds
                
                self.bounds = Bounds(lb = lower.ravel(), ub = upper.ravel())
                           
    
    def _compute_acq(self, x, return_terms=False):
        
        mean_p, std_p = self.gp_model.predict(X = x.reshape(1,-1), return_std = True)
        
#        original Unravel -> self.delta = np.random.randn() # Moved to next_x
        
        t1 = self.weight[0] * -np.linalg.norm(x - self.X_init - self.std_x * self.delta / np.log(self.iter))

        t2 = self.weight[1] * std_p
        
        f_acqu = t1 + t2
                                         
        if return_terms:
            return f_acqu, t1, t2
        else:
            return f_acqu

    def _compute_acq_inverse(self, x, return_terms=False):
        
        return -self._compute_acq(x, return_terms=False)
        

    def next_x(self, gp_model):
        
        self.iter     = self.iter + 1
        self.delta    = np.random.randn()
        self.gp_model = gp_model

        if self.sampling:
            self.X_next = self.next_x_sampling(gp_model)
            
        else:
            self.X_next =  self.next_x_opt(gp_model)
            
        self.X_next = self.X_next.reshape(1,-1)
            
        print('NEXT X: ', self.X_next)
        return self.X_next
            
        
    def next_x_sampling(self, gp_model):
        
        self.X_diffs = np.empty([self.n_samples+1])
        self.acqu_fs = np.empty([self.n_samples+1])
        
        X_next        = self.perturbed_X(self.X_init)
        max_acq_value = self._compute_acq(X_next)
        
        self.X_diffs[0] = 0
        self.acqu_fs[0] = max_acq_value
        
        
        for sample in range(self.n_samples):
            
            x = self.X_init + self.std_x * self.sampled_dist[sample,:]
        
            self.X_diffs[sample+1] = self.Calculate_X0_Distance(x)

            acq_value = self._compute_acq(x)

            self.acqu_fs[sample+1] = acq_value
            
            if acq_value > max_acq_value:
                max_acq_value = acq_value
                X_next        = x
                #print(X_next)

                
        #self.Plot_X_Diff_Acq()
               
        return X_next
                
    
    def next_x_opt(self, gp_model):

        print('Bounds: ', self.bounds)
        opt_result = minimize(fun    = self._compute_acq_inverse,
                              x0     = self.X_init,
                              method = self.opt_method,
                              bounds = self.bounds)
        
        print(opt_result)
        if opt_result.success:
            X_next = opt_result.x
            
        else:
            X_next = self.perturbed_X(self.X_next)
       
        return X_next
                
    
    def Set_Weight(self, weight):
        
        if isinstance(weight, float):
            self.weight = [weight, 1.0 - weight]
            
        else:
            self.weight = weight            


    def perturbed_X(self, X, perb=0.25):
        return X + self.std_x * (np.random.random() * perb - perb)
        
        
    def Calculate_X0_Distance(self, X):
        
        distance_sq = np.square((self.X_init - X)/self.std_x)
    
        distance_RMS = np.sqrt(np.mean(distance_sq))
        
        return distance_RMS
        
        
    def Plot_X_Diff_Acq(self):
        
        plt.scatter(self.X_diffs, self.acqu_fs)
    

    