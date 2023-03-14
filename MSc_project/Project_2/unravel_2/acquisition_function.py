# Import Libraries
import numpy as np

class Acq_Base(object):

    def __init__(self, X_init):
        pass
    
    def next_x(self):
        pass
    
    
    
class FUR_W(Acq_Base):
    
    def __init__(self, X_init, std_x, weight=None):
        
        self.X_init = X_init
        
        self.iter = 1
        
        self.sample = 0 ###################################################################################################
        
        self.std_x = std_x
        
        self.n_features = X_init.shape[1]
        
        if weight == None:
            self.weight = [0.5, 0.5]
        
        elif isinstance(weight, float):
            self.weight = [weight, 1.0 - weight]
            
        else:
            self.weight = weight           
    

    def acq_function(self, x):
        
        mean_p, std_p = self.gp_model.predict(X = x, return_std = True)
        
        self.delta = np.random.randn()#self.n_features)
        
        f_acqu = self.weight[0] * -np.linalg.norm(x - self.X_init - self.std_x * self.delta / np.log(self.iter)) + \
                 self.weight[1] * std_p
        
        if self.sample % 5 == 0:
            print(self.iter, ' : ', x, ' : ', mean_p,' : ', std_p, ' : ', f_acqu)

        x_diff = (x - self.X_init) / self.std_x
        
        x_diff = np.sqrt(np.mean(np.square(x_diff), axis=1))
        
        self.debug_data[self.sample,:] = np.hstack([x_diff, mean_p, std_p, f_acqu])##########################################################
        #self.debug_data[self.sample,:] = np.array([1,2,3,4])
        
        
                                   
        return f_acqu
    

    def next_x(self, gp_model, n_samples=20): ##########################n_samples
        
        self.iter = self.iter + 1
        
        self.gp_model = gp_model

        self.debug_data = np.empty([n_samples+1,4])
        self.sample = 0


        min_acq_value = self.acq_function(self.X_init)
        x_next        = self.X_init
        
        for sample in range(n_samples):
            
            self.sample = sample + 1 ###################################################################################################

            x = self.X_init + self.std_x * np.random.randn(self.n_features)
        
            acq_value = self.acq_function(x)
            
            if acq_value < min_acq_value:
                min_acq_value = acq_value
                x_next = x
                #print(x_next)
                
        return x_next, self.debug_data
                
    
    def Set_Weight(weight):
        
        if isinstance(weight, float):
            self.weight = [weight, 1.0 - weight]
            
        else:
            self.weight = weight            


    
    
    
    
    
    