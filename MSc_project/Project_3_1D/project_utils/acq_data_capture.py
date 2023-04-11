import numpy as np
import matplotlib.pyplot as plt


class Acq_Data(object):
    
    color_list = ['midnightblue', 'navy', 'darkblue', 'mediumblue', 'blue',                   \
                  'slateblue', 'mediumslateblue', 'mediumpurple', 'blueviolet', 'darkviolet', \
                  'mediumvioletred', 'crimson', 'deeppink', 'red', 'magenta']#,                 \
                  #'coral', 'darkorange', 'orange', 'gold', 'yellow']

    def __init__(self): pass
        
    def new_X(self, X, y, fe_x0, acq_function, t1_t2=False): pass

    def plot_point(self, p=0): pass
        
    def plot_all(self): pass


###############################################################################################################################        
    
class Acq_Data_1D(Acq_Data):

    def __init__(self):
        
        self.num_acq_points = 100
        
        self.norm_x_range = np.arange(0.0, 1.0, 1.0/self.num_acq_points)
        self.X_range      = np.empty([self.num_acq_points, 1])
        self.X_range[:,0] = self.norm_x_range
        
        self.X_values   = np.empty([0,1])
        self.y_values   = np.empty([0,1])
        self.acq_values = np.empty([0,self.num_acq_points])
        self.t1         = np.empty([0,self.num_acq_points])
        self.t2         = np.empty([0,self.num_acq_points])

        self.fe_x0 = float("NaN")

        self.N_iter_points = 0
        
        
    def new_X(self, X, y, fe_x0, acq_function, t1_t2=False):
        
        #print('X ', X)
        
        self.X_values = np.vstack([self.X_values, X.ravel()])
        self.y_values = np.vstack([self.y_values, y.ravel()])
        
        acq_values   = np.empty([self.num_acq_points])
        t1           = np.empty([self.num_acq_points])
        t2           = np.empty([self.num_acq_points])
    
        for i in range(self.num_acq_points):
            if t1_t2:
                acq_values[i], t1[i], t2[i] = acq_function._compute_acq(self.X_range[i].reshape(1,-1), return_terms=True)
            else:
                acq_values[i] = acq_function._compute_acq(self.X_range[i].reshape(1,-1))
                
                
        
        self.acq_values = np.vstack([self.acq_values, acq_values])
        self.t1         = np.vstack([self.t1,         t1])
        self.t2         = np.vstack([self.t2,         t2])
        
        self.fe_x0 = fe_x0

        
        self.N_iter_points = self.N_iter_points + 1

        
    @staticmethod        
    def Forrester(x):
        return np.square(6.0*x - 2.0)*np.sin(12.0*x - 4.0)
            
    def plot_point(self, p=0):
        
        if p < 0 or p >= self.N_iter_points:
            print("Out of Range Point")
            return
        
        
        yvals  = Acq_Data_1D.Forrester(self.norm_x_range)
        
        fig, ax1 = plt.subplots()
        
        color = 'darkblue'
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax1.plot(self.norm_x_range, yvals, linewidth=1.0, color = color)
        
        color = 'green'
        ax1.scatter([self.X_values[p]], [self.y_values[p]], linewidth=1.0, color = color, marker = 'o')
        
        color = 'red'
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        ax2.set_ylabel('Acquisition Function')
        ax2.plot(self.norm_x_range, self.acq_values[p], color=color,       label='FUR W')
        ax2.plot(self.norm_x_range, self.t1[p],         color='lime',      label='T1')
        ax2.plot(self.norm_x_range, self.t2[p],         color='darkgreen', label='T2')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend()


        fig.tight_layout()
        
        plt.show()
                        
        
    def plot_all(self):
        
        yvals  = Acq_Data_1D.Forrester(self.norm_x_range)
       
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 10]})
        
        ax1.set_yticks(ticks=[])
        #ax1.set_xticks(ticks=[0,5,10,15,20])
        ax1.set_xlim([0, self.N_iter_points+1])
        ax1.set_xlabel('Iteration Number')
        
        color = 'darkblue'
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax2.plot(self.norm_x_range, yvals, linewidth=1.0, color = color)

        for point in range(self.N_iter_points):
            
            col_idx = int(float(point * len(Acq_Data.color_list)) / float(self.N_iter_points))
            color   = Acq_Data.color_list[col_idx]
            
            ax2.scatter([self.X_values[point]], [self.y_values[point]], color = color, marker = 'o')
        
            ax1.scatter(x=[point+1], y=[0], color = color, marker = 'o')
            
        fig.tight_layout()

        plt.show()
                        
    def get_fe_x0(self):
        return self.fe_x0

###############################################################################################################################        
    
class Acq_Data_2D(Acq_Data):

    def __init__(self):
        
        self.num_acq_points = 100
        
        self.norm_x_range = np.arange(0.0, 1.0, 1.0/self.num_acq_points)
        self.X_range      = np.empty([self.num_acq_points, 1])
        self.X_range[:,0] = self.norm_x_range
        
        self.X_values   = np.empty([0,1])
        self.y_values   = np.empty([0,1])
        self.acq_values = np.empty([0,self.num_acq_points])
        self.t1         = np.empty([0,self.num_acq_points])
        self.t2         = np.empty([0,self.num_acq_points])

        self.fe_x0 = float("NaN")

        self.N_iter_points = 0
        
        
    def new_X(self, X, y, fe_x0, acq_function, t1_t2=False):
        return
        #print('X ', X)
        
        self.X_values = np.vstack([self.X_values, X.ravel()])
        self.y_values = np.vstack([self.y_values, y.ravel()])
        
        acq_values   = np.empty([self.num_acq_points])
        t1           = np.empty([self.num_acq_points])
        t2           = np.empty([self.num_acq_points])
    
        for i in range(self.num_acq_points):
            if t1_t2:
                acq_values[i], t1[i], t2[i] = acq_function._compute_acq(self.X_range[i].reshape(1,-1), return_terms=True)
            else:
                acq_values[i] = acq_function._compute_acq(self.X_range[i].reshape(1,-1))
                
                
        
        self.acq_values = np.vstack([self.acq_values, acq_values])
        self.t1         = np.vstack([self.t1,         t1])
        self.t2         = np.vstack([self.t2,         t2])
        
        self.fe_x0 = fe_x0

        
        self.N_iter_points = self.N_iter_points + 1

        
    @staticmethod        
    def Forrester(x):
        return np.square(6.0*x - 2.0)*np.sin(12.0*x - 4.0)
            
    def plot_point(self, p=0):
        return
        
        if p < 0 or p >= self.N_iter_points:
            print("Out of Range Point")
            return
        
        
        yvals  = Acq_Data_1D.Forrester(self.norm_x_range)
        
        fig, ax1 = plt.subplots()
        
        color = 'darkblue'
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax1.plot(self.norm_x_range, yvals, linewidth=1.0, color = color)
        
        color = 'green'
        ax1.scatter([self.X_values[p]], [self.y_values[p]], linewidth=1.0, color = color, marker = 'o')
        
        color = 'red'
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        ax2.set_ylabel('Acquisition Function')
        ax2.plot(self.norm_x_range, self.acq_values[p], color=color,       label='FUR W')
        ax2.plot(self.norm_x_range, self.t1[p],         color='lime',      label='T1')
        ax2.plot(self.norm_x_range, self.t2[p],         color='darkgreen', label='T2')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend()


        fig.tight_layout()
        
        plt.show()
                        
        
    def plot_all(self):
        return
        yvals  = Acq_Data_1D.Forrester(self.norm_x_range)
       
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 10]})
        
        ax1.set_yticks(ticks=[])
        #ax1.set_xticks(ticks=[0,5,10,15,20])
        ax1.set_xlim([0, self.N_iter_points+1])
        ax1.set_xlabel('Iteration Number')
        
        color = 'darkblue'
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax2.plot(self.norm_x_range, yvals, linewidth=1.0, color = color)

        for point in range(self.N_iter_points):
            
            col_idx = int(float(point * len(Acq_Data.color_list)) / float(self.N_iter_points))
            color   = Acq_Data.color_list[col_idx]
            
            ax2.scatter([self.X_values[point]], [self.y_values[point]], color = color, marker = 'o')
        
            ax1.scatter(x=[point+1], y=[0], color = color, marker = 'o')
            
        fig.tight_layout()

        plt.show()
                        
    def get_fe_x0(self):
        return self.fe_x0

###############################################################################################################################


class Acq_Data_nD(object):

    def __init__(self, X_Init, bounds, BB_Model=None):
        
        self.num_acq_points   = 300
        self.normalised_range = 1.5
        
        self.X_Init     = X_Init
        self.N_features = X_Init.size
        self.bounds     = np.array(bounds).reshape(2, self.N_features)
        self.BB_Model   = BB_Model

        self.norm_x_range = np.arange(-self.normalised_range, self.normalised_range, \
                                      2.0 * self.normalised_range /self.num_acq_points)
        
        self.X_range = np.empty([self.num_acq_points, self.N_features])
        
        self.bounds_range = (self.bounds[1,:] - self.bounds[0,:]) / 2.0
        self.bounds_mean  = (self.bounds[0,:] + self.bounds[1,:]) / 2.0
        
        for idx, xval in enumerate(self.norm_x_range):
            self.X_range[idx,:] = self.bounds_mean + xval * self.bounds_range
            
        self.X_values   = np.empty([0,self.N_features])
        self.y_values   = np.empty([0,1])
        self.acq_values = np.empty([0,self.num_acq_points])
        self.t1         = np.empty([0,self.num_acq_points])
        self.t2         = np.empty([0,self.num_acq_points])

        self.fe_x0 = float("NaN")

        self.N_iter_points = 0
        
        
        
        
    def new_X(self, X, y, fe_x0, acq_function, t1_t2=False):
        
        self.X_values = np.vstack([self.X_values, X.ravel()])
        self.y_values = np.vstack([self.y_values, y.ravel()])
        
        acq_values   = np.empty([self.num_acq_points])
        t1           = np.empty([self.num_acq_points])
        t2           = np.empty([self.num_acq_points])
    
        for i in range(self.num_acq_points):
            if t1_t2:
                acq_values[i], t1[i], t2[i] = acq_function._compute_acq(self.X_range[i].reshape(1,-1), return_terms=True)
            else:
                acq_values[i] = acq_function._compute_acq(self.X_range[i].reshape(1,-1))
                
                
        
        self.acq_values = np.vstack([self.acq_values, acq_values])
        self.t1         = np.vstack([self.t1,         t1])
        self.t2         = np.vstack([self.t2,         t2])
        
        self.fe_x0 = fe_x0

        
        self.N_iter_points = self.N_iter_points + 1

        
    def Add_BB_model(self, BB_Model):
        self.BB_Model = BB_Model
            

    def BB_model_prediction(self, x):
        return self.BB_Model.predict(x)
    
    def Normalise_X(self, X):
        return np.mean((X - self.bounds_mean) / self.bounds_range)
            

    def Create_BB_plot(self):
        
        self.BB_predictions = np.empty([self.num_acq_points])

        for idx in range(self.num_acq_points):
            self.BB_predictions[idx] = self.BB_Model.predict(self.X_range[idx].reshape(1,-1))
            #print(self.X_range[idx], ' < X y > ', self.BB_predictions[idx])


    def Add_BB_plot(self, ax, color):
      
        ax.plot(self.norm_x_range, self.BB_predictions, linewidth=1.0, color = color)        

    def plot_point(self, p=0):

        if p < 0 or p >= self.N_iter_points:
            print("Out of Range Point")
            return
        
        fig, ax1 = plt.subplots()
      
        color = 'darkblue'
        
        ax1.set_xlabel('X - Normalised Average')
        ax1.set_xlim([-1.05 * self.normalised_range, 1.05 * self.normalised_range])
        #ax1.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax1.set_ylabel('F(X)  - Black Box Model')
        ax1.tick_params(axis='y', labelcolor=color)

        self.Add_BB_plot(ax = ax1, color = color)
        
        normalised_X = self.Normalise_X(self.X_values[p])
        color = 'green'
        ax1.scatter([normalised_X], [self.y_values[p]], linewidth=1.0, color = color, marker = 'o')
       
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        

        color = 'red'
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylabel('Acquisition Function')
        ax2.plot(self.norm_x_range, self.acq_values[p], color=color,       label='FUR W')
        ax2.plot(self.norm_x_range, self.t1[p],         color='lime',      label='T1')
        ax2.plot(self.norm_x_range, self.t2[p],         color='darkgreen', label='T2')
        ax2.legend()


        fig.tight_layout()
        
        plt.show()
                        
        
        
    def plot_all(self):
               
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 10]})
              
        ax1.set_yticks(ticks=[])
        ax1.set_xlim([0, self.N_iter_points+1])
        #ax1.set_xticks(ticks=[0,5,10,15,20])
        ax1.set_xlabel('Iteration Number')
        
        color = 'darkblue'        
        
        ax2.set_xlabel('X - Normalised Average')
        ax2.set_xlim([-1.05 * self.normalised_range, 1.05 * self.normalised_range])
        #ax2.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax2.set_ylabel('F(X) - Black Box Model')
        ax2.tick_params(axis='y', labelcolor=color)

        self.Add_BB_plot(ax = ax2, color = color)

        for point in range(self.N_iter_points):
            
            col_idx = int(float(point * len(Acq_Data.color_list)) / float(self.N_iter_points))
            color   = Acq_Data.color_list[col_idx]
            
            normalised_X = self.Normalise_X(self.X_values[point])
            ax2.scatter([normalised_X], [self.y_values[point]], color = color, marker = 'o')
        
            ax1.scatter(x=[point+1], y=[0], color = color, marker = 'o')
            
        fig.tight_layout()

        plt.show()
                            
    
    def get_fe_x0(self):
        return self.fe_x0


    # for debug
    def get_X_BB(self):
    
        return self.BB_predictions, self.X_range

        
