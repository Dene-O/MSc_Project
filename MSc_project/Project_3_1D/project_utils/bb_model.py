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

from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
#from sklearn.datasets import load_boston

import matplotlib.pyplot as plt



class BB_Model(object):

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
                 random_state=None):

        self.data_frame = None
        
        if dataset == 'Boston':

            self.mode = 'regression'
            
            self.feature_names = ['crime_rate','zoned_lots','industry','by_river','NOX','avg_rooms','pre_1940',
                                 'emp_distance','rad_access','tax_rate','pupil_tea_rat','low_status']

            self.outcome = 'median_value'
            
            self.catagorical_features = [3, 8]
            
            self.Read_File('datasets/boston_adj.csv')
            
            
        elif dataset == 'Wine':

            self.mode = mode
            
            self.feature_names = ['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides',
                                  'free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']

            self.outcome              = 'quality'
            self.class_names          = ['3','4','5','6','7','8','9']
            self.catagorical_features = []
            
            self.Read_File('datasets/winequality-white.csv')
            
        elif dataset == 'Diabetes':

            self.mode = 'classification'
            
            self.feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin',
                                  'BMI','DiabetesPedigreeFunction','Age']

            self.outcome = 'Outcome'
            
            self.class_names = ['Healthy', 'Diabetic']
            
            self.catagorical_features = []

            self.Read_File('datasets/diabetes.csv')
            
            
          
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
                
            print(feature_names)
                
            for feature in range(n_passive, n_features):                
                feature_names.append('Active_' + str(feature))

            print(feature_names)
                            
            coeff_order = np.argsort(coeffs)

            print('Coeffs and Order: ',coeffs, coeff_order)
                
            self.feature_names = []
            for feature_index in range(n_features):       
                for coeff_index in range(n_features):
                    if feature_index == coeff_order[coeff_index]:
                        self.feature_names.append(feature_names[coeff_index])
                        break
            
            print(self.feature_names)
            print(coeff_order)
            print(coeffs)
            
            self.outcome = 'Y_Value'
            
                 
            
        elif dataset == 'Forrester' or dataset == 'forrester':
        
            self.mode = 'regression'
            self.catagorical_features = []
            
            n_features = 1
            
            X = np.arange(-0.05, 0.95, 0.001)
#            X = np.arange(0.28, 0.52, 0.001)
            self.y = BB_Model.Forrester(X)

            print(X.shape)
            self.feature_names = ['X']
            self.outcome = 'Forrester'

            fig, ax = plt.subplots()
            ax.plot(X, self.y, linewidth=1.0)
            plt.show()
                        
            self.X = np.empty([1000, 1])
#            self.X = np.empty([240, 1])
            
            self.X[:,0] = X
            
            print(self.X.shape)

            
        elif dataset == 'Forrester_2D' or dataset == 'forrester_2d':
        
            self.mode = 'regression'
            self.catagorical_features = []
            
            n_features = 2
            N_x1       = 10
            N_x2       = 10
            N_x_all    = N_x1 * N_x2
            
            x1_range = np.arange(0.0, 1.0, 1.0/N_x1)
            x2_range = np.arange(0.0, 1.0, 1.0/N_x2)

            self.X      = np.empty([N_x_all, 2])
            self.y      = np.empty (N_x_all)
            self.y_plot = np.empty([N_x1, N_x2])
            
            idx = 0
            for idx1 in range(N_x1):
                for idx2 in range(N_x2):
                    
                    self.X[idx, 0] = x1_range[idx1]
                    self.X[idx, 1] = x2_range[idx2]                    
                    self.y[idx]    = BB_Model.Forrester_2D(self.X[idx,:])
                    idx += 1
            

            print('X shape',self.X.shape)
            print('y shape',self.y.shape)

            self.feature_names = ['X1', 'X2']
            self.outcome = 'Forrester 2D'
            
            
            X1, X2 = np.meshgrid(x1_range, x2_range)
            
            Z = BB_Model.Forrester(X1) + BB_Model.Forrester(X2)

            fig, ax = plt.subplots()
            
            ax.contour(X1, X2, Z)
            
            plt.show()
                                   
            

            
        else:
            self.mode = mode

            self.feature_names        = feature_names
            self.outcome              = outcome
            self.class_names          = classes
            self.catagorical_features = catagorical_features

            self.Read_File(dataset)
            

            
        self.random_state = check_random_state(random_state)
           
        self.X_train, self.X_test, self.y_train, self.y_test = \
          train_test_split(self.X, self.y, train_size=train_size, random_state=self.random_state)
        

    def Read_File(self, filename):
        
        self.data_frame = pd.read_csv(filename)

        self.y = np.array(self.data_frame[self.outcome])
        self.X = np.array(self.data_frame[self.feature_names])
        

                          
    # used in debugging only
    def df(self):
        if self.data_frame == None:
            print('None Dataframe for this Dataset')
        else:
            return self.data_frame
            
    @staticmethod        
    def Forrester(x):
        return np.square(6.0*x - 2.0)*np.sin(12.0*x - 4.0)
            
    @staticmethod        
    def Forrester_2D(X):
        return BB_Model.Forrester(X[0]) + BB_Model.Forrester(X[1]) 
            
    def MPL(self, show_score=True):
    
        if self.mode == 'regression' or self.mode == 'Regression':
            self.MPL_model = MLPRegressor(random_state=self.random_state)
            
        else:
            self.MPL_model = MLPClassifier(random_state=self.random_state)

        self.MPL_model.fit(self.X_train, self.y_train)
        
        if show_score:
            print(self.MPL_model.score(self.X_test, self.y_test))
            
        return self.MPL_model
    
    
    def Random_Forest(self, show_score=True):
    
        if self.mode == 'regression' or self.mode == 'Regression':
            self.RF_model = RandomForestRegressor(random_state=self.random_state)
            
        else:
            self.RF_model = RandomForestClassifier(random_state=self.random_state)
            
        self.RF_model.fit(self.X_train, self.y_train)
        
        if show_score:
            print(self.RF_model.score(self.X_test, self.y_test))
            
        return self.RF_model
    
    
    def L_Regression(self, show_score=True):
    
        if self.mode == 'regression' or self.mode == 'Regression':
            self.LR_model = LinearRegression()
            
        else:
            self.LR_model = LogisticRegression()
            
        self.LR_model.fit(self.X_train, self.y_train)
        
        if show_score:
            print(self.LR_model.score(self.X_test, self.y_test))
            
        return self.LR_model
    
    def GP(self, show_score=True):
    
        if self.mode == 'regression' or self.mode == 'Regression':
            self.GP_model = GaussianProcessRegressor()
            
        else:
            self.GP_model = GaussianProcessClassifier()
            
        self.GP_model.fit(self.X_train, self.y_train)
        
        if show_score:
            print(self.GP_model.score(self.X_test, self.y_test))
            
        return self.GP_model
    
    
    
    
    
    def get_mode(self):
        return self.mode
    
    def get_TT_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_catagorical(self):
        return self.catagorical_features
    
    def get_features(self):
        return self.feature_names
    
    def get_classes(self):
        return self.class_names
    
    def get_MPL(self):
        return self.MPL_model
    
    def get_Random_Forest(self):
        return self.RF_model
    
    def get_L_Regression(self, show_score=True):           
        return self.LR_model  
   
    def get_GP(self, show_score=True):           
        return self.GP_model
    
   
    
    
    
    
    