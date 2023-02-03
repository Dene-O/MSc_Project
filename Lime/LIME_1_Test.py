# LIME 1 Test
# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from lime import lime_tabular


# Load Data

PK_data = pd.read_csv("parkinsons.csv")

features = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", \
            "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", \
            "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]

#features_subset = np.random.choice(features, size=6)
features_subset = features


#Select Data

y = PK_data["status"]
X = PK_data[features_subset]

# Splitting X & y into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80)


#Train and Test a Random Forest with the data

PK_Model = RandomForestClassifier(n_estimators=100, criterion='log_loss')
PK_Model.fit(X_train, y_train)

print('Model Score', PK_Model.score(X_test, y_test))

# LIME Explanation

PK_Explain = lime_tabular.LimeTabularExplainer(training_data=np.array(X_train),
                                               mode="classification",
                                               training_labels=None,
                                               feature_names=features_subset)
                                               #categorical_features=None,
                                               #categorical_names=None,
                                               #kernel_width=None,
                                               #kernel=None,
                                               #verbose=False,
                                               #class_names=None,
                                               #feature_selection='auto',
                                               #discretize_continuous=True,
                                               #discretizer='quartile',
                                               #sample_around_instance=False,
                                               #random_state=None,
                                               #training_data_stats=None


# Looks at Explanation

PK_explain_inst = PK_Explain.explain_instance(data_row=X_test.iloc[inst],
                                              predict_fn=PK_Model.predict_proba,
                                              #labels=features_subset)
                                              #top_labels=None,
                                              num_features=5)
                                              #num_samples=5000,
                                              #distance_metric='euclidean',
                                              #model_regressor=None,
                                              #sampling_method='gaussian')

print(PK_explain_inst.as_list())