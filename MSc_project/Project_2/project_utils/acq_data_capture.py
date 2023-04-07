import numpy as np

import matplotlib.pyplot as plt


class Acq_Data(object):

    color_list = ['midnightblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue',    \
                  'mediumslateblue', 'mediumpurple', 'blueviolet', 'darkviolet', 'magenta', \
                  'mediumvioletred', 'deeppink', 'crimson', 'red']

    def __init__(self):
        
        self.num_acq_points = 100
        
        self.X_values   = np.empty([0,1])
        self.y_values   = np.empty([0,1])
        self.acq_values = np.empty([0,self.num_acq_points])

        self.fe_x0 = float("NaN")

        self.N_points = 0

        
    def new_X(self, X, y, fe_x0, acq_function):        
        
        self.X_values = np.vstack([self.X_values, X.ravel()])
        self.y_values = np.vstack([self.y_values, y.ravel()])
        
        xrange       = np.arange(0.0, 1.0, 1.0/self.num_acq_points)
        X_range      = np.empty([self.num_acq_points, 1])
        X_range[:,0] = xrange
        acq_values   = np.empty([self.num_acq_points])
    
#        acq_values = acq_function._compute_acq(X_range)       
        for i in range(self.num_acq_points):
            acq_values[i] = acq_function._compute_acq(X_range[i])

        
        self.acq_values = np.vstack([self.acq_values, acq_values])
        
        self.fe_x0 = fe_x0

        
        self.N_points = self.N_points + 1

        
    @staticmethod        
    def Forrester(x):
        return np.square(6.0*x - 2.0)*np.sin(12.0*x - 4.0)
            
    def plot_point(self, p=0):
        
        if p < 0 or p >= self.N_points:
            print("Out of Range Point")
            return
        
        
        xrange = np.arange(0.0, 1.0, 1.0/self.num_acq_points)
        yvals  = Acq_Data.Forrester(xrange)
        
        fig, ax1 = plt.subplots()
        
        color = 'darkblue'
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax1.plot(xrange, yvals, linewidth=1.0, color = color)
        
        color = 'green'
        ax1.scatter([self.X_values[p]], [self.y_values[p]], linewidth=1.0, color = color, marker = 'o')
        
        color = 'red'
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        
        ax2.set_ylabel('Acquisition Function')
        ax2.plot(xrange, self.acq_values[p], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        
        plt.show()
                        
        
    def plot_all(self):
        
        xrange = np.arange(0.0, 1.0, 1.0/self.num_acq_points)
        yvals  = Acq_Data.Forrester(xrange)
       
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 10]})
        
        color = 'darkblue'
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax2.plot(xrange, yvals, linewidth=1.0, color = color)

        for point in range(self.N_points):
            index = int(float(point * len(Acq_Data.color_list)) / self.N_points)
            color = Acq_Data.color_list[index]
            ax2.scatter([self.X_values[point]], [self.y_values[point]], color = color, marker = 'o')
        
            ax1.scatter(x=[point+1], y=[0], color = color, marker = 'o')
            
        ax1.set_yticks(ticks=[])
        ax1.set_xticks(ticks=[0,5,10,15,20])
        ax1.set_xlabel('Iteration Number')
        
        fig.tight_layout()

        plt.show()
                        
    def get_fe_x0(self):
        return self.fe_x0

        