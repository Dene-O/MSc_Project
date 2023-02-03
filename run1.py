
import numpy as np
import pandas as pd
from sklearn import preprocessing

import Parkinson



def Import_Parkinsons():

    PK_data = pd.read_csv('Datasets/parkinsons.csv')

    x_names = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',\
               'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',\
               'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ',\
               'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

    for name in x_names:
  
        vector = PK_data[[name]]

        y_values = np.array(PK_data[['status']])

        scaler = preprocessing.MinMaxScaler()
        vector_scaled = scaler.fit_transform(vector)
    
        if (name == x_names[0]):
            x_values = vector_scaled
        else:
            x_values = np.hstack((x_values,vector_scaled))

    return x_values, y_values



x_values, y_values = Import_Parkinsons()

p = Parkinson()

print(x_values.shape)
print(y_values.shape)