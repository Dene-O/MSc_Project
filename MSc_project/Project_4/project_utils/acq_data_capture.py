import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.qmc import LatinHypercube

from project_utils.bb_model import BB_Model


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

    def __init__(self, X_Init, bounds, BB_Model=None):

        print('Acq_Data_1D')
        
       
        self.num_acq_points   = 100
        self.normalised_range = 1.5
        
        self.X_Init     = X_Init
        self.N_features = 1
        self.bounds     = np.array(bounds).ravel()
        self.BB_Model   = BB_Model

        self.bounds_range = (self.bounds[1] - self.bounds[0]) / 2.0
        self.bounds_mean  = (self.bounds[0] + self.bounds[1]) / 2.0
        
        
        self.norm_x_range = np.arange(self.bounds_mean - self.normalised_range * self.bounds_range, \
                                      self.bounds_mean + self.normalised_range * self.bounds_range, \
                                      self.bounds_range * self.normalised_range * 2 / self.num_acq_points)
        
        self.num_acq_points = self.norm_x_range.size
        
        self.X_range = np.empty([self.num_acq_points,1])
        
        for idx in range(self.num_acq_points):
            self.X_range[idx,0] = self.norm_x_range[idx]
            
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

        
    def Add_BB_model(self, BB_Model):
        self.BB_Model = BB_Model

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
        ax1.set_xlabel('x')
        ax1.set_ylabel('F(X)  - Black Box Model')
        ax1.tick_params(axis='y', labelcolor=color)
        
        self.Add_BB_plot(ax = ax1, color = color)
        
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
              
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 10]})
        
        ax1.set_yticks(ticks=[])
        ax1.set_xlim([0, self.N_iter_points+1])
        ax1.set_xlabel('Iteration Number')
        
        color = 'darkblue'
        ax2.set_xlabel('X - Iterations')
        ax1.set_ylabel('y - Iterations')
        ax2.tick_params(axis='y', labelcolor=color)
        
        self.Add_BB_plot(ax = ax2, color = color)
        
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
    
class Acq_Data_1D_For(Acq_Data):

    def __init__(self):
        
        print('Acq_Data_1D_For')
        
        self.num_acq_points = 100
        
        self.norm_x_range = np.arange(0.0, 1.0, 1.0/self.num_acq_points)
        self.X_range      = np.empty([self.num_acq_points, 1])
        self.X_range[:,0] = self.norm_x_range
        
        self.X_values   = np.empty([0,1])
        self.y_values   = np.empty([0,1])
        self.acq_values = np.empty([0,self.num_acq_points])
        self.t1         = np.empty([0,self.num_acq_points])
        self.t2         = np.empty([0,self.num_acq_points])

        self.min_acq_data = np.empty([0, 3])
        print('SHAPE0: ',self.min_acq_data.shape)
        
        self.fe_x0 = float("NaN")

        self.N_iter_points = 0
        
        
    def new_X(self, X, y, fe_x0, acq_function, t1_t2=False):
        
        #print('X ', X)
        
        print('ITER:', self.N_iter_points)
        self.X_values = np.vstack([self.X_values, X.ravel()])
        self.y_values = np.vstack([self.y_values, y.ravel()])
        
        acq_values   = np.empty([self.num_acq_points])
        t1           = np.empty([self.num_acq_points])
        t2           = np.empty([self.num_acq_points])
    
        for i in range(self.num_acq_points):
            if t1_t2:
                acq_values[i], t1[i], t2[i] = acq_function._compute_acq(self.X_range[i].reshape(1,-1), return_terms=True)
                if abs(np.mean(X) - np.mean(self.X_range[i])) < 0.015:
                    #print('XX: ', X, self.X_range[i])
                    #print('MIN:', acq_values[i], t1[i], t2[i])
                    min_acqxx, min_t1xx, min_t2xx = acq_function._compute_acq(self.X_range[i].reshape(1,-1), return_terms=True)
            else:
                acq_values[i] = acq_function._compute_acq(self.X_range[i].reshape(1,-1))
                
                
        
        self.acq_values = np.vstack([self.acq_values, acq_values])
        self.t1         = np.vstack([self.t1,         t1])
        self.t2         = np.vstack([self.t2,         t2])
        
        min_acq_data = acq_function._compute_acq(X, return_terms=True)
        
        min_acq_data = np.array(min_acq_data).reshape(1,3)
        
        self.min_acq_data = np.vstack([self.min_acq_data, min_acq_data])
        
        
        self.fe_x0 = fe_x0
      
        self.N_iter_points = self.N_iter_points + 1

        
    @staticmethod        
    def Forrester(x):
        return np.square(6.0*x - 2.0)*np.sin(12.0*x - 4.0)
            
    def plot_point(self, p=0):
        
        if p < 0 or p >= self.N_iter_points:
            print("Out of Range Point")
            return
        
        
        yvals  = Acq_Data_1D_For.Forrester(self.norm_x_range)
        
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
        
        yvals  = Acq_Data_1D_For.Forrester(self.norm_x_range)
       
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
    
    def Add_BB_model(self, BB_Model):
        self.BB_Model = BB_Model



    def plot_t1_t2(self, point):
        
        fig, ax = plt.subplots()
        
        ax.set_xlabel('t1')
        ax.set_ylabel('t2')
        
        t1 = self.t1[point,:]
        t2 = self.t2[point,:]

        ax.plot(t1, t2, color = 'darkblue')
            
        ax.scatter(self.min_acq_data[point,1], self.min_acq_data[point,2], color = 'green', marker = 'D')
        #print('IN plot', self.min_acq_data[point,1:2])
        
                    
        fig.tight_layout()

        plt.show()
        
###############################################################################################################################        
    
class Acq_Data_2D(Acq_Data):

    def __init__(self, X_Init, bounds, BB_Model=None):

        print('Acq_Data_2D')
        
        ## START MATCH BB MODEL ##
        
        self.N_x1    = 100
        self.N_x2    = 100
        self.N_x_all = self.N_x1 * self.N_x2
        
        self.normalised_range = 1.25
        
        self.X_Init     = X_Init
        self.N_features = 2
        self.BB_Model   = BB_Model
        self.bounds     = bounds

        self.bounds_range = (self.bounds[1,:] - self.bounds[0,:]) / 2.0
        self.bounds_mean  = (self.bounds[0,:] + self.bounds[1,:]) / 2.0       

        x1_range = np.arange(self.bounds_mean[0]  - self.normalised_range * self.bounds_range[0], \
                             self.bounds_mean[0]  + self.normalised_range * self.bounds_range[0], \
                             self.bounds_range[0] * self.normalised_range * 2 / self.N_x1)
        
        x1_range = x1_range[0:self.N_x1]

        x2_range = np.arange(self.bounds_mean[1]  - self.normalised_range * self.bounds_range[1], \
                             self.bounds_mean[1]  + self.normalised_range * self.bounds_range[1], \
                             self.bounds_range[1] * self.normalised_range * 2 / self.N_x2)
        
        x2_range = x2_range[0:self.N_x2]

            
        print('x1_range size',x1_range.size)
        print('x2_range size',x2_range.size)
        
        
        self.X = np.empty([self.N_x_all, 2])
        self.y = np.empty (self.N_x_all)
        
        idx = 0
        for idx1 in range(self.N_x1):
            for idx2 in range(self.N_x2):
                
                self.X[idx, 0] = x1_range[idx1]
                self.X[idx, 1] = x2_range[idx2]                   
                idx += 1
                                      
            
        self.X1, self.X2 = np.meshgrid(x1_range, x2_range)
        print('self.X1 ', self.X1.size)
        print('self.X2 ', self.X2.size)
                
        self.X_values   = np.empty([0,2])
        self.y_values   = np.empty([0,1])
        self.acq_values = np.empty([0, self.N_x1, self.N_x2])
        self.t1         = np.empty([0, self.N_x1, self.N_x2])
        self.t2         = np.empty([0, self.N_x1, self.N_x2])

        self.fe_x0 = float("NaN")

        self.N_iter_points = 0
        
        
    def new_X(self, X, y, fe_x0, acq_function, t1_t2=False):

        #print('X ', X)
        
        self.X_values = np.vstack([self.X_values, X.ravel()])
        self.y_values = np.vstack([self.y_values, y.ravel()])
        
        acq_values   = np.empty([self.N_x1, self.N_x2])
        t1           = np.empty([self.N_x1, self.N_x2])
        t2           = np.empty([self.N_x1, self.N_x2])
    
        for idx1 in range(self.N_x1):
            for idx2 in range(self.N_x2):
                if t1_t2:
                    acq_values[idx1, idx2], t1[idx1, idx2], t2[idx1, idx2] = acq_function._compute_acq(self.X[self.N_x2 * idx1 + idx2].reshape(1,-1), return_terms=True)
                else:
                    acq_values[idx1, idx2] = acq_function._compute_acq(self.X[self.N_x2 * idx1 + idx2].reshape(1,-1))
                
      
        self.acq_values = np.vstack([self.acq_values, acq_values.reshape(1, self.N_x1, self.N_x2)])
        self.t1         = np.vstack([self.t1,                 t1.reshape(1, self.N_x1, self.N_x2)])
        self.t2         = np.vstack([self.t2,                 t2.reshape(1, self.N_x1, self.N_x2)])
        
        #print('acq_values', self.acq_values.shape, acq_values.shape)       

        self.fe_x0 = fe_x0

        
        self.N_iter_points = self.N_iter_points + 1

        
            
    def Add_BB_model(self, BB_Model):
        self.BB_Model = BB_Model

    def Create_BB_plot(self):
        
        self.BB_predictions = np.empty([self.N_x1, self.N_x2])

        print('IDX 0 -',  self.N_x1-1, ':', end=' ')

        for idx1 in range(self.N_x1):
            print(idx1, end=' ')
            for idx2 in range(self.N_x2):
                
                X = np.array([self.X1[idx1, idx2], self.X2[idx1, idx2]]).reshape(1,-1)
                
                self.BB_predictions[idx1, idx2] = self.BB_Model.predict(X)
                
        print(', done!')


    
    def Two_D_plot(self, ax, levels, fontsize):
        
        contours = ax.contour(self.X1, self.X2, self.BB_predictions, levels = levels)
        ax.clabel(contours, inline=True, fontsize = fontsize)
            
            
    def plot_point(self, point, plot_4=True):
        
        if point < 0 or point >= self.N_iter_points:
            print("Out of Range Point")
            return
        

        if plot_4:
            fig, axs = plt.subplots(2,2)
        
            color = 'red'
            for idx1 in range(2):
                for idx2 in range(2):
                    axs[idx1,idx2].scatter([self.X_values[point,0]], [self.X_values[point,1]], color = color, marker = 'D')
        
            self.Two_D_plot(axs[0,0], 8, 8)
        
            axs[0,0].set_xticks(ticks=[])
            
            contours = axs[0,1].contour(self.X1, self.X2, self.acq_values[point], levels = 8)
            axs[0,1].clabel(contours, inline=True, fontsize=8)
            axs[0,1].set_xticks(ticks=[])
            axs[0,1].set_yticks(ticks=[])
            
            contours = axs[1,0].contour(self.X1, self.X2, self.t1[point], levels = 8)
            axs[1,0].clabel(contours, inline=True, fontsize=6)
            
            contours = axs[1,1].contour(self.X1, self.X2, self.t2[point], levels = 8)
            axs[1,1].clabel(contours, inline=True, fontsize=8)
            axs[1,1].set_yticks(ticks=[])
            
        else: #Single Plot
            fig, ax = plt.subplots()
        
            color = 'red'
            ax.scatter([self.X_values[point,0]], [self.X_values[point,1]], color = color, marker = 'D')
            
            contours = ax.contour(self.X1, self.X2, self.acq_values[point], levels = 10)
            ax.clabel(contours, inline=True, fontsize=10)
            
        fig.tight_layout()
        
        plt.show()
                        
        
    def plot_all(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 10]})

        ax1.set_yticks(ticks=[])
        #ax1.set_xticks(ticks=[0,5,10,15,20])
        ax1.set_xlim([0, self.N_iter_points+1])
        ax1.set_xlabel('Iteration Number')
        
        self.Two_D_plot(ax2, 8, 8)

        for point in range(self.N_iter_points):
            
            col_idx = int(float(point * len(Acq_Data.color_list)) / float(self.N_iter_points))
            color   = Acq_Data.color_list[col_idx]
            
            ax1.scatter(x=[point+1], y=[0], color = color, marker = 'o')
            ax2.scatter([self.X_values[point,0]], [self.X_values[point,1]], color = color, marker = 'o')
        
            
        fig.tight_layout()

        plt.show()
                        
    def get_fe_x0(self):
        return self.fe_x0

###############################################################################################################################        
    
class Acq_Data_2D_For(Acq_Data):

    def __init__(self):

        print('Acq_Data_2D_For')
        
        ## START MATCH BB MODEL ##
        
        n_features = 2
        self.N_x1       = 100
        self.N_x2       = 100
        self.N_x_all    = self.N_x1 * self.N_x2
            
        x1_range = np.arange(0, 1.0, 1.0/self.N_x1)
        x2_range = np.arange(0, 1.0, 1.0/self.N_x2)
            

        self.X      = np.empty([self.N_x_all, n_features])
        self.y      = np.empty (self.N_x_all)
        #self.y_plot = np.empty([self.N_x1, self.N_x2])
        
        idx = 0
        for idx1 in range(self.N_x1):
            for idx2 in range(self.N_x2):
                
                self.X[idx, 0] = x1_range[idx1]
                self.X[idx, 1] = x2_range[idx2]                
                idx += 1
               
            
        self.X1, self.X2 = np.meshgrid(x1_range, x2_range)
        
        ## END MATCH BB MODEL ##
        
        self.X_values   = np.empty([0, n_features])
        self.y_values   = np.empty([0, 1])
        self.acq_values = np.empty([0, self.N_x1, self.N_x2])
        self.t1         = np.empty([0, self.N_x1, self.N_x2])
        self.t2         = np.empty([0, self.N_x1, self.N_x2])

        self.fe_x0 = float("NaN")

        self.N_iter_points = 0
        
        
    def new_X(self, X, y, fe_x0, acq_function, t1_t2=False):

        #print('X ', X)
        
        self.X_values = np.vstack([self.X_values, X.ravel()])
        self.y_values = np.vstack([self.y_values, y.ravel()])
        
        acq_values   = np.empty([self.N_x1, self.N_x2])
        t1           = np.empty([self.N_x1, self.N_x2])
        t2           = np.empty([self.N_x1, self.N_x2])
    
        for idx1 in range(self.N_x1):
            for idx2 in range(self.N_x2):
                if t1_t2:
                    acq_values[idx1, idx2], t1[idx1, idx2], t2[idx1, idx2] = acq_function._compute_acq(self.X[self.N_x2 * idx1 + idx2].reshape(1,-1), return_terms=True)
                else:
                    acq_values[idx1, idx2] = acq_function._compute_acq(self.X[self.N_x2 * idx1 + idx2].reshape(1,-1))
                
      
        self.acq_values = np.vstack([self.acq_values, acq_values.reshape(1, self.N_x1, self.N_x2)])
        self.t1         = np.vstack([self.t1,                 t1.reshape(1, self.N_x1, self.N_x2)])
        self.t2         = np.vstack([self.t2,                 t2.reshape(1, self.N_x1, self.N_x2)])
        
        #print('acq_values', self.acq_values.shape, acq_values.shape)       

        self.fe_x0 = fe_x0

        
        self.N_iter_points = self.N_iter_points + 1

        
            
    def plot_point(self, point, plot_4=True):
        
        if point < 0 or point >= self.N_iter_points:
            print("Out of Range Point")
            return
        

        if plot_4:
            fig, axs = plt.subplots(2,2)
        
            color = 'red'
            for idx1 in range(2):
                for idx2 in range(2):
                    axs[idx1,idx2].scatter([self.X_values[point,0]], [self.X_values[point,1]], color = color, marker = 'D')
                    axs[idx1,idx2].set_xlim(0, 1)
                    axs[idx1,idx2].set_ylim(0, 1)

        
            self.BB_Model.Forrester_plot_2D(axs[0,0], 8, 8)
        
            axs[0,0].set_xticks(ticks=[])
            
            contours = axs[0,1].contour(self.X1, self.X2, self.acq_values[point], levels = 8)
            axs[0,1].clabel(contours, inline=True, fontsize=8)
            axs[0,1].set_xticks(ticks=[])
            axs[0,1].set_yticks(ticks=[])
            
            contours = axs[1,0].contour(self.X1, self.X2, self.t1[point], levels = 8)
            axs[1,0].clabel(contours, inline=True, fontsize=6)
            
            contours = axs[1,1].contour(self.X1, self.X2, self.t2[point], levels = 8)
            axs[1,1].clabel(contours, inline=True, fontsize=8)
            axs[1,1].set_yticks(ticks=[])
            
        else: #Single Plot
            fig, ax = plt.subplots()
        
            color = 'red'
            ax.scatter([self.X_values[point,0]], [self.X_values[point,1]], color = color, marker = 'D')
            
            ax.set_xlim(0, 10.0)
            ax.set_ylim(0, 20.0)
       
            contours = ax.contour(self.X1, self.X2, self.acq_values[point], levels = 10)
            ax.clabel(contours, inline=True, fontsize=10)
            
        fig.tight_layout()
        
        plt.show()
                        
        
    def plot_all(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 10]})

        ax1.set_yticks(ticks=[])
        #ax1.set_xticks(ticks=[0,5,10,15,20])
        ax1.set_xlim([0, self.N_iter_points+1])
        ax1.set_xlabel('Iteration Number')
        
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
       
        self.BB_Model.Two_D_plot(ax2, 8, 8)

        for point in range(self.N_iter_points):
            
            col_idx = int(float(point * len(Acq_Data.color_list)) / float(self.N_iter_points))
            color   = Acq_Data.color_list[col_idx]
            
            ax1.scatter(x=[point+1], y=[0], color = color, marker = 'o')
            ax2.scatter([self.X_values[point,0]], [self.X_values[point,1]], color = color, marker = 'o')
        
            
        fig.tight_layout()

        plt.show()
                        
    def get_fe_x0(self):
        return self.fe_x0

    def Add_BB_model(self, BB_Model):
        self.BB_Model = BB_Model

###############################################################################################################################


class Acq_Data_nD(Acq_Data):

    def __init__(self, X_Init, bounds, BB_Model=None):
        
        print('Acq_Data_nD')
        
       
        self.num_acq_points   = 400
        self.normalised_range = 1
        
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
        self.acq_values = np.empty([0,self.num_acq_points,2])
        self.t1         = np.empty([0,self.num_acq_points,2])
        self.t2         = np.empty([0,self.num_acq_points,2])

        self.min_acq_data = np.empty([0, 3])
        self.X_0_acq_data = np.empty([0, 3])
        
        self.fe_x0 = float("NaN")

        self.N_iter_points = 0
        
        
    def new_X(self, X, y, fe_x0, acq_function, t1_t2=False):
        
        self.X_values = np.vstack([self.X_values, X.ravel()])
        self.y_values = np.vstack([self.y_values, y.ravel()])
        
        acq_values   = np.empty([1,self.num_acq_points,2])
        t1           = np.empty([1,self.num_acq_points,2])
        t2           = np.empty([1,self.num_acq_points,2])

        # Use a Latin Hyper cube
        sampler = LatinHypercube(d = self.N_features, strength = 1)

        sampled_dist = sampler.random(n = self.num_acq_points) * self.normalised_range - (self.normalised_range / 2)
        
        sampled_dist = sampled_dist * self.bounds_range + self.bounds_mean
    
        for i in range(self.num_acq_points):
            if t1_t2:
                acq_values[0,i,0], t1[0,i,0], t2[0,i,0] = acq_function._compute_acq(self.X_range[i,:].reshape(1,-1), return_terms=True)
                acq_values[0,i,1], t1[0,i,1], t2[0,i,1] = acq_function._compute_acq(sampled_dist[i,:].reshape(1,-1), return_terms=True)
            else:
                acq_values[0,i,0] = acq_function._compute_acq(self.X_range[i,:].reshape(1,-1), return_terms=True)
                acq_values[0,i,1] = acq_function._compute_acq(sampled_dist[i,:].reshape(1,-1))
                
                
        
        self.acq_values = np.vstack([self.acq_values, acq_values])
        self.t1         = np.vstack([self.t1,         t1])
        self.t2         = np.vstack([self.t2,         t2])
        
        
        min_acq_data = acq_function._compute_acq(X, return_terms=True)

        self.min_acq_data =np.vstack([self.min_acq_data, min_acq_data])
        
        X_0_acq_data = acq_function._compute_acq(self.X_Init, return_terms=True)

        self.X_0_acq_data =np.vstack([self.X_0_acq_data, X_0_acq_data])
        
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
        ax2.plot(self.norm_x_range, self.acq_values[p,:,0], color=color,       label='FUR W')
        ax2.plot(self.norm_x_range, self.t1[p,:,0],         color='lime',      label='T1')
        ax2.plot(self.norm_x_range, self.t2[p,:,0],         color='darkgreen', label='T2')
 
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
                            
    def plot_t1_t2(self, point):
        
        fig, ax = plt.subplots()
        
        ax.set_xlabel('t1')
        ax.set_ylabel('t2')
        
        ax.plot(self.t1[point,:,0], self.t2[point,:,0], color = 'green', linewidth = 2, label = 'SD Rng')
        ax.scatter(self.t1[point,:,1], self.t2[point,:,1], color = 'darkblue',  marker = '.', label = 'LHC')
        
        ax.scatter(self.min_acq_data[point,1], self.min_acq_data[point,2], color = 'orange', marker = 'D', label = 'X Next')
        ax.scatter(self.X_0_acq_data[point,1], self.X_0_acq_data[point,2], color = 'red',    marker = 'D', label = 'X Init')
                    
        fig.tight_layout()

        ax.legend()
        plt.show()
        
    
    def get_fe_x0(self):
        return self.fe_x0


    # for debug
    def get_X_BB(self):
    
        return self.BB_predictions, self.X_range

        
