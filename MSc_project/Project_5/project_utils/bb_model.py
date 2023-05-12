import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state

from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
#from sklearn.datasets import load_boston

import matplotlib.pyplot as plt


class BB_Model(object):

    ####################################################################################################################
    def __init__(self,
                 dataset,
                 mode='classification',
                 feature_names=None,
                 catagorical_features=None,
                 outcome=None,
                 classes=None,
                 train_size=0.80,
                 N_samples=500,
                 Feature_Counts=[],
                 random_state=None,
                 plot_file=""):

        self.data_frame = None

        ################################################################################################################
        if dataset == 'Boston':

            self.mode = 'regression'
            
            self.feature_names = ['crime_rate','zoned_lots','industry','by_river','NOX','avg_rooms','pre_1940',
                                 'emp_distance','rad_access','tax_rate','pupil_tea_rat','low_status']

            self.outcome = 'median_value'
            
            self.catagorical_features = [3, 8]
            
            self.Read_File('datasets/boston_adj.csv')
            
            
        ################################################################################################################
        elif dataset == 'Wine':

            self.mode = mode
            
            self.feature_names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
                                  'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

            self.outcome              = 'quality'
            self.class_names          = ['3','4','5','6','7','8','9']
            self.catagorical_features = []
            
            self.Read_File('datasets/winequality-white.csv')
            
        ################################################################################################################
        elif dataset == 'Diabetes':

            self.mode = 'classification'
            
            self.feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
                                  'BMI','DiabetesPedigreeFunction','Age']

            self.outcome = 'Outcome'
            
            self.class_names = ['Healthy', 'Diabetic']
            
            self.catagorical_features = []

            self.Read_File('datasets/diabetes.csv')
            
            
          
        ################################################################################################################
        elif dataset == 'Classification' or dataset == 'classification':
        
            self.mode = 'classification'
            self.catagorical_features = []
            
            n_features    = Feature_Counts[0]
            n_informative = Feature_Counts[1]
            
            self.X, self.y = make_classification(n_samples     = N_samples,
                                                 n_features    = n_features,
                                                 n_informative = n_informative,
                                                 n_classes     = 2,
                                                 random_state  = random_state)
          
            self.outcome = 'Outcome'

            
        ################################################################################################################
        elif dataset == 'Regression' or dataset == 'regression':
            
            self.mode = 'regression'
            self.catagorical_features = []
            
            n_features    = Feature_Counts[0]
            n_informative = Feature_Counts[1]
            n_passive     = n_features - n_informative
            
            self.X, self.y, coeffs = make_regression(n_samples     = N_samples,
                                                     n_features    = n_features,
                                                     n_informative = n_informative,
                                                     n_targets     = 1,
                                                     noise         = 1.0,
                                                     coef          = True,
                                                     random_state  = random_state)
 
            feature_names = []
    
            for feature in range(n_passive):                
                feature_names.append('Passive_' + str(feature))
                
               
            for feature in range(n_passive, n_features):                
                feature_names.append('Active_' + str(feature))

                           
            coeff_order = np.argsort(coeffs)

            #print('Coeffs and Order: ',coeffs, coeff_order)
                
            self.feature_names = []
            for feature_index in range(n_features):       
                for coeff_index in range(n_features):
                    if feature_index == coeff_order[coeff_index]:
                        self.feature_names.append(feature_names[coeff_index])
                        break
            
            #print('Features:    ', self.feature_names)
            #print('Coeff Order: ', coeff_order)
            #print('Coeffs:      ', coeffs)
            
            self.outcome = 'Y_Value'
            
                 
            
        ################################################################################################################
        elif dataset == 'Synthetic_Reg' or dataset == 'synthetic_reg':
            
            self.mode = 'regression'
            self.catagorical_features = []
            
            n_features    = 8
            n_informative = 6
            n_passive     = n_features - n_informative
            
            self.feature_names = []

            Feature_Means = np.random.uniform(low=5.0, high=20.0, size=n_features)
            Feature_SDs   = np.random.uniform(low=1.0, high=5.0,  size=n_features)

            self.X = np.empty([N_samples, n_features])

            # Features 0, 1: Passive uniform distribution
            for feature in range(2):
                
                self.feature_names.append('Passive_' + str(feature))
                
                lower_range = Feature_Means[feature] - 2 * Feature_SDs[feature]
                upper_range = Feature_Means[feature] + 2 * Feature_SDs[feature]
                
                self.X[:,feature] = np.random.uniform(low=lower_range, high=upper_range, size=N_samples)
                
            # Features 2, 3, 4: Active uniform distribution
            for feature in range(2, 5):  
                
                self.feature_names.append('Active_' + str(feature))

                lower_range = Feature_Means[feature] - 2 * Feature_SDs[feature]
                upper_range = Feature_Means[feature] + 2 * Feature_SDs[feature]
                
                if lower_range < 0: lower_range = 0
                
                self.X[:,feature] = np.random.uniform(low=lower_range, high=upper_range, size=N_samples)
                
            # Features 5, 6, 7: Active n dormalistribution
            sign = 1
            for feature in range(5, 8): 
                
                self.feature_names.append('Active_' + str(feature))
                
                while (Feature_Means[feature] / 2) < Feature_SDs[feature]:
                    Feature_SDs[feature] = Feature_SDs[feature] / 2
               
                self.X[:,feature] =  sign * np.random.normal(Feature_Means[feature], Feature_SDs[feature], size=N_samples)
                
                sign = -sign

                

            self.Feature_Coeffs = np.zeros([N_samples, n_features])

            self.Feature_Coeffs[:,2] = np.random.normal(5,  2,    size=N_samples)
            self.Feature_Coeffs[:,3] = np.random.normal(-8, 1,    size=N_samples)
            self.Feature_Coeffs[:,4] = np.random.normal(-4, 2,    size=N_samples)
            self.Feature_Coeffs[:,5] = np.random.normal(10, 2,    size=N_samples)
            self.Feature_Coeffs[:,6] = np.random.normal(6,  3,    size=N_samples)
            self.Feature_Coeffs[:,7] = np.random.normal(-2, 0.25, size=N_samples)
            
            self.y = np.sum(self.X * self.Feature_Coeffs, axis = 1)
            
                 
            
        ################################################################################################################
        elif dataset == 'Forrester' or dataset == 'forrester':
        
            self.mode = 'regression'
            self.catagorical_features = []
            
            n_features = 1
            
            X = np.arange(-0.05, 0.95, 0.001)
#            X = np.arange(0.28, 0.52, 0.001)
            self.y = BB_Model.Forrester(X)

            #print(X.shape)
            self.feature_names = ['X']
            self.outcome = 'Forrester'

            fig, ax = plt.subplots()
            ax.plot(X, self.y, linewidth=1.0)

            if plot_file != "":
                plt.savefig(fname=plot_file)
        
            plt.show()
                        
            self.X = np.empty([1000, 1])
#            self.X = np.empty([240, 1])
            
            self.X[:,0] = X
            
            #print(self.X.shape)

            
        ################################################################################################################
        elif dataset == 'Forrester_2D' or dataset == 'forrester_2d':
        
            self.mode = 'regression'
            self.catagorical_features = []
            
            n_features = 2
            N_x1       = 100
            N_x2       = 100
            N_x_all    = N_x1 * N_x2
            
            x1_range = np.arange(0.0, 1.0, 1.0/N_x1)
            x2_range = np.arange(0.0, 1.0, 1.0/N_x2)

            self.X      = np.empty([N_x_all, n_features])
            self.y      = np.empty (N_x_all)
            #self.y_plot = np.empty([N_x1, N_x2])
            
            idx = 0
            for idx1 in range(N_x1):
                for idx2 in range(N_x2):
                    
                    self.X[idx, 0] = x1_range[idx1]
                    self.X[idx, 1] = x2_range[idx2]                    
                    self.y[idx]    = BB_Model.Forrester_2D(self.X[idx,:])
                    idx += 1
            

            #print('X shape',self.X.shape)
            #print('y shape',self.y.shape)

            self.feature_names = ['X1', 'X2']
            self.outcome = 'Forrester 2D'
            
            
            self.X1, self.X2 = np.meshgrid(x1_range, x2_range)
            
            self.Z = BB_Model.Forrester(self.X1) + BB_Model.Forrester(self.X2)

            fig, ax = plt.subplots()
            
            contours = ax.contour(self.X1, self.X2, self.Z, levels = 8)
            ax.clabel(contours, inline=True, fontsize=10)
            
            if plot_file != "":
                plt.savefig(fname=plot_file)
        
            plt.show()
            

        ################################################################################################################
        elif dataset == 'Two_D' or dataset == 'two_d':
        
            self.mode = 'regression'
            self.catagorical_features = []
            
            n_features = 2
            N_x1       = 100
            N_x2       = 100
            N_x_all    = N_x1 * N_x2
            
            x1_range = np.arange(0, 10.0, 10.0/N_x1)
            x2_range = np.arange(1, 20.0, 19.0/N_x2)
            
            y1 = np.empty(N_x1)
            y2 = np.empty(N_x2)

            for idx, x1 in enumerate(x1_range):
                y1[idx] = 12 - (x1-3)*(x1-3)

            for idx, x2 in enumerate(x2_range):
                y2[idx] = 2 * x2 + 8 * np.sin(2 * x2) / x2
            
            #print('x1_range: ', np.mean(x1_range), np.std(x1_range))
            #print('Y1: ', np.mean(y1), np.std(y1))
            #print('x2_range: ', np.mean(x2_range), np.std(x2_range))
            #print('Y2: ', np.mean(y2), np.std(y2))
            
            self.X = np.empty([N_x_all, n_features])
            self.y = np.empty (N_x_all)
            self.Z = np.empty([N_x1, N_x2])
            
            self.X1, self.X2 = np.meshgrid(x1_range, x2_range)
            
            idx = 0
            for idx1 in range(N_x1):
                for idx2 in range(N_x2):
                    
                    self.X[idx, 0] = x1_range[idx1]
                    self.X[idx, 1] = x2_range[idx2]  
                    
                    self.y[idx] = y1[idx1] + y2[idx2]
                    
                    self.Z[idx1, idx2] = y1[idx1] + y2[idx2]
                           
                    idx += 1
            

            print('X shape',self.X.shape)
            print('y shape',self.y.shape)

            self.feature_names = ['X1', 'X2']
            self.outcome = 'Regression 2D'
            
            
            #fig, (ax1, ax2) = plt.subplots(2, 1)
            #ax1.plot(x1,y1)
            #ax2.plot(x2,y2)
            
            fig, ax = plt.subplots()
            
            contours = ax.contour(self.X1, self.X2, self.Z)
            ax.clabel(contours, inline=True, fontsize=10)
            
            if plot_file != "":
                plt.savefig(fname=plot_file)
        
            plt.show()
            

            
        ################################################################################################################
        else:
            self.mode = mode

            self.feature_names        = feature_names
            self.outcome              = outcome
            self.class_names          = classes
            self.catagorical_features = catagorical_features

            self.Read_File(dataset)
            
        ################################################################################################################

            
        self.random_state = check_random_state(random_state)
           
        self.X_train, self.X_test, self.y_train, self.y_test = \
          train_test_split(self.X, self.y, train_size=train_size, random_state=self.random_state)
        
        self.X_train_std = np.std(self.X_train, axis = 0)
        

    ####################################################################################################################
    def Read_File(self, filename):
        
        self.data_frame = pd.read_csv(filename)

        self.y = np.array(self.data_frame[self.outcome])
        self.X = np.array(self.data_frame[self.feature_names])
        

                          
    ####################################################################################################################
    # used in debugging only
    def df(self):
        if self.data_frame == None:
            print('None Dataframe for this Dataset')
        else:
            return self.data_frame
            
    ####################################################################################################################
    @staticmethod        
    def Forrester(x):
        return np.square(6.0*x - 2.0)*np.sin(12.0*x - 4.0)
            
    @staticmethod        
    def Forrester_2D(X):
        return BB_Model.Forrester(X[0]) + BB_Model.Forrester(X[1]) 
            
    ####################################################################################################################
    def Forrester_plot_2D(self, ax, levels, fontsize):
        
        contours = ax.contour(self.X1, self.X2, self.Z, levels = levels)
        ax.clabel(contours, inline=True, fontsize = fontsize)
            
            
            
    ####################################################################################################################
    def Two_D_plot(self, ax, levels, fontsize):
        
        contours = ax.contour(self.X1, self.X2, self.Z, levels = levels)
        ax.clabel(contours, inline=True, fontsize = fontsize)
            
            
            
    ####################################################################################################################
    def MPL(self, show_score=True):
    
        if self.mode == 'regression' or self.mode == 'Regression':
            self.MPL_model = MLPRegressor(random_state=self.random_state)
            
        else:
            self.MPL_model = MLPClassifier(random_state=self.random_state)

        self.MPL_model.fit(self.X_train, self.y_train)
        
        if show_score:
            print(self.MPL_model.score(self.X_test, self.y_test))
            
        return self.MPL_model
    
    
    ####################################################################################################################
    def Random_Forest(self, show_score=True):
    
        if self.mode == 'regression' or self.mode == 'Regression':
            self.RF_model = RandomForestRegressor(random_state=self.random_state)
            
        else:
            self.RF_model = RandomForestClassifier(random_state=self.random_state)
            
        self.RF_model.fit(self.X_train, self.y_train)
        
        if show_score:
            print(self.RF_model.score(self.X_test, self.y_test))
            
        return self.RF_model
    
    
    ####################################################################################################################
    def L_Regression(self, show_score=True):
    
        if self.mode == 'regression' or self.mode == 'Regression':
            self.LR_model = LinearRegression()
            
        else:
            self.LR_model = LogisticRegression()
            
        self.LR_model.fit(self.X_train, self.y_train)
        
        if show_score:
            print(self.LR_model.score(self.X_test, self.y_test))
            
        return self.LR_model
    
    ####################################################################################################################
    def GP(self, show_score=True):
    
        if self.mode == 'regression' or self.mode == 'Regression':
            self.GP_model = GaussianProcessRegressor()
            
        else:
            self.GP_model = GaussianProcessClassifier()
            
        self.GP_model.fit(self.X_train, self.y_train)
        
        if show_score:
            print(self.GP_model.score(self.X_test, self.y_test))
            
        return self.GP_model
    
    
    
    ####################################################################################################################
    def get_mode(self):
        return self.mode
    
    ####################################################################################################################
    def get_TT_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    ####################################################################################################################
    def get_catagorical(self):
        return self.catagorical_features
    
    ####################################################################################################################
    def get_features(self):
        return self.feature_names
    
    ####################################################################################################################
    def get_classes(self):
        return self.class_names
    
    ####################################################################################################################
    def get_MPL(self):
        return self.MPL_model
    
    ####################################################################################################################
    def get_Random_Forest(self):
        return self.RF_model
    
    ####################################################################################################################
    def get_L_Regression(self, show_score=True):           
        return self.LR_model  
   
    ####################################################################################################################
    def get_GP(self, show_score=True):           
        return self.GP_model
    
    ####################################################################################################################
    def get_2d_data(self):
        return self.X, self.X1, self.X2
    
    ####################################################################################################################
    def get_Feature_Coeffs(self):
        return np.mean(self.Feature_Coeffs, axis = 0)
    
    ####################################################################################################################
    def get_X_train_std(self):
        return self.X_train_std
    
    
    