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

        
    
class Acq_Data_1D(Acq_Data):

    def __init__(self):
        
        self.num_acq_points = 100
        
        self.X_values   = np.empty([0,1])
        self.y_values   = np.empty([0,1])
        self.acq_values = np.empty([0,self.num_acq_points])
        self.t1         = np.empty([0,self.num_acq_points])
        self.t2         = np.empty([0,self.num_acq_points])

        self.fe_x0 = float("NaN")

        self.N_points = 0
        
        
    def new_X(self, X, y, fe_x0, acq_function, t1_t2=False):
        
        #print('X ', X)
        
        self.X_values = np.vstack([self.X_values, X.ravel()])
        self.y_values = np.vstack([self.y_values, y.ravel()])
        
        xrange       = np.arange(0.0, 1.0, 1.0/self.num_acq_points)
        X_range      = np.empty([self.num_acq_points, 1])
        X_range[:,0] = xrange
        acq_values   = np.empty([self.num_acq_points])
        t1           = np.empty([self.num_acq_points])
        t2           = np.empty([self.num_acq_points])
    
#        acq_values = acq_function._compute_acq(X_range)
        for i in range(self.num_acq_points):
            if t1_t2:
                acq_values[i], t1[i], t2[i] = acq_function._compute_acq(X_range[i].reshape(1,-1), return_terms=True)
                #if i==10 or i==25 or i==50 or i==75 or i==90:
                #    print('ZZ: ', X_range[i], acq_values[i], t1[i], t2[i], (acq_values[i] - t1[i] - t2[i])) 
            else:
                acq_values[i] = acq_function._compute_acq(X_range[i].reshape(1,-1))
                
                
        
        self.acq_values = np.vstack([self.acq_values, acq_values])
        self.t1         = np.vstack([self.t1,         t1])
        self.t2         = np.vstack([self.t2,         t2])
        
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
        yvals  = Acq_Data_1D.Forrester(xrange)
        
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
        ax2.plot(xrange, self.acq_values[p], color=color,       label='FUR W')
        ax2.plot(xrange, self.t1[p],         color='lime',      label='T1')
        ax2.plot(xrange, self.t2[p],         color='darkgreen', label='T2')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend()


        fig.tight_layout()
        
        plt.show()
                        
        
    def plot_all(self):
        
        xrange = np.arange(0.0, 1.0, 1.0/self.num_acq_points)
        yvals  = Acq_Data_1D.Forrester(xrange)
       
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

        