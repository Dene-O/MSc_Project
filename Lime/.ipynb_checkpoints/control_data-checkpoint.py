import numpy as np

from sklearn.utils import check_random_state


class Control_Data(object):

    def __init__(self,
                 Y_Name="Y_Value",
                 Y_std_dev=None,
                 Y_Func=None,
                 Y_classes=None,
                 Active_X=[],
                 Passive_X=[],
                 Num_Samples=50,
                 feature_bounds=None,
                 random_state=None):

        ###############################################################################
        # get the number of inputs and raise an error if there are none.
        self.Num_X = len(Active_X) + len(Passive_X)
        
        if self.Num_X == 0:
            raise NotImplementedError("No input columns specified")

        # set number of columns and numer of active columns     
        self.Num_X        = len(Active_X) + len(Passive_X)
        self.Num_Active_X = len(Active_X)

        ###############################################################################
        # Assign feature names
        if Active_X == None:
            self.Active_X = list(range(self.Num_Active_X))
        else:
            self.Active_X = Active_X
        
        if Passive_X == None:
            self.Passive_X = list(range(self.Num_Active_X, self.Num_X))
        else:
            self.Passive_X = Passive_X
        
        self.Feature_Labels = self.Active_X + self.Passive_X

        ###############################################################################
        self.Num_Samples = Num_Samples
        

        ###############################################################################
        #Assign the Feature bounds, ranges and means
        
        # when feature_bounds is None set all ranges to 0-1 
        if feature_bounds == None:
            zeros = np.zeros(self.Num_X, dtype=float)
            ones  = np.ones(self.Num_X, dtype=float)
            self.feature_bounds = np.column_stack((zeros, ones))
            
        elif np.size(np.array(feature_bounds).ravel()) == 2:
            
            lowers = np.ones(self.Num_X, dtype=float) * feature_bounds[0]
            uppers = np.ones(self.Num_X, dtype=float) * feature_bounds[1]
            self.feature_bounds = np.column_stack((lowers, uppers))

        # otherwise assume feature_bounds contains the ranges of all the features
        else:
            self.feature_bounds = np.array(feature_bounds)
        
        #print('feature_bounds: ', self.feature_bounds)
        
        # mean and range of features
        self.feature_means = np.empty(self.Num_Active_X, dtype=float)
        self.feature_range = np.empty(self.Num_Active_X, dtype=float)
        
        for feature in range(self.Num_Active_X):
            self.feature_means[feature] = (self.feature_bounds[feature, 1] + self.feature_bounds[feature, 0]) / 2.0
            self.feature_range[feature] =  self.feature_bounds[feature, 1] - self.feature_bounds[feature, 0]
        
               
        ###############################################################################
        self.random_state = check_random_state(random_state)

        ###############################################################################
        # assign the feature values
        self.X = np.empty([self.Num_Samples, self.Num_X], dtype=float)

        for feature in range(self.Num_X):
            self.X[:,feature] = self.random_state.uniform(self.feature_bounds[feature,0], \
                                                          self.feature_bounds[feature,1], Num_Samples)

        #print('X: ', self.X)

        ###############################################################################
        # assign Y values, class names and function
        
        if Y_Name == None:
            self.Y_Name = "Y_Value"
        else:
            self.Y_Name = Y_Name
            
            
        self.Y_classes = Y_classes

        # assign the function that generates Y values from the feature values
        
        if Y_Func == None or Y_Func == 'Regression':
            
            if Y_std_dev == None:
                self.Y_std_dev = 3.0               
            else:
                self.Y_std_dev = Y_std_dev
            
            self.Y_Func = self.Default_Regression    
            
        elif Y_Func == None or Y_Func == 'Classification':
            
            if Y_std_dev == None:
                self.Y_std_dev = 1.0                
            else:
                self.Y_std_dev = Y_std_dev
            
                self.Y_Func = self.Default_Classification
            
            if self.Y_classes == None:
                self.Y_classes = ['Zero', 'One']
        
        elif Y_Func == None or Y_Func == 'Regression_2':
            
            if Y_std_dev == None:
                self.Y_std_dev = 4.0               
            else:
                self.Y_std_dev = Y_std_dev
            
            self.Y_Func = self.Regression_2
            
        else:
            self.Y_Func = Y_Func

    
        self.Y = np.empty(self.Num_Samples, dtype=float)

        for row in range(self.Num_Samples):
            self.Y[row] = self.Y_Func(row = self.X[row,:])
                                     
        #print('Y: ', self.Y)
        
        ###############################################################################

        

    def Get_Features(self):
        return self.X
    
    
    def Get_Feature_Names(self):
        return self.Feature_Labels
        
    def Get_Outcomes(self):
        return self.Y
        
    def Get_Class_Names(self):
        return self.Y_classes
        
    def Predict(self, data_row):
        return self.Y_Func(row = data_row)
        

    # This function returns the sum of the feature values multipled by their column indices
    def Default_Regression(self, row):

        result = self.random_state.normal(0, self.Y_std_dev, 1)[0]
        
        for feature in range(self.Num_Active_X): 
            result += (feature + 10) * row[feature]
            
        return result
    
            
    # This function returns the sum of the feature values multipled by a fluctuating seed value
    def Regression_2(self, row, seed=1, seed_limit=3, seed_multiplier=1.25):

        result = self.random_state.normal(0, self.Y_std_dev, 1)[0]
        
        for feature in range(self.Num_Active_X): 
            result += seed * row[feature]
            seed   = seed * seed_multiplier * -1
            
            if abs(seed) > seed_limit:
                seed = seed / seed_limit           
            
        return result
    
    # This  function returns 0 if the sum of scaled (between -0.5 to 0.5) active features is less
    # than 0. If the sum is above 0 return 1.
    def Default_Classification(self, row):
        
        result = self.random_state.normal(0, self.Y_std_dev, 1)[0]
        
        for feature in range(self.Num_Active_X): 
            result += (row[feature] - self.feature_means[feature]) / self.feature_range[feature]
            
        if result < 0: result = 0
        else:          result = 1
        
        return result
    
    
    