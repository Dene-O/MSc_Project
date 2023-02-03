# Parkinson Class
# Loads and stores Parkinson data

import numpy as np
import pandas as pd
from sklearn import preprocessing

class Parkinson:
    
    def __init__(self):
        
        PK_data = pd.read_csv('Datasets/parkinsons.csv')

        self.x_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',\
                        'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',\
                        'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ',\
                        'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

        self.y_values = np.array(PK_data[['status']])


        for name in self.x_names:
  
            vector = PK_data[[name]]

            scaler = preprocessing.MinMaxScaler()
            vector_scaled = scaler.fit_transform(vector)
    
            if (name == self.x_names[0]):
                x_values = vector_scaled
            else:
                 x_values = np.hstack((x_values,vector_scaled))

        self.x_values = x_values


    def x_vals(self):
        return self.x_values

    def y_vals(self):
        return self.y_values

    def x_y_values(self):
        return self.x_values, self.y_values







