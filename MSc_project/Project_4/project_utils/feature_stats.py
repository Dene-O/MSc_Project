import numpy as np
import matplotlib.pyplot as plt

from project_utils.uncertainty_plot import Add_Uncertainty_Plot

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from copy import deepcopy

from itertools import combinations

import pickle

#from sklearn.utils import check_random_state


class Feature_Statistics(object):

    #list of colours and its range for dipslaying graphs
    colour_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:yellow']
    colour_max  = len(colour_list)
                          
    def __init__(self,
                 feature_names=None,
                 mode='classification',
                 classes=None, 
                 X_train_std=1, 
                 uncert_pr=True, 
                 N_consistancy=0,
                 file_handle=None):

        if file_handle == None:
            
            self.Feature_Names  = feature_names
            self.Num_Features   = len(feature_names)
            self.Num_Samples    = 0
            self.Feature_Scores = np.empty([0, self.Num_Features], dtype=float)
            self.Scaled_Scores  = np.empty([0, self.Num_Features], dtype=float)
            self.Features       = np.empty([0, self.Num_Features], dtype=float)
            self.Exp_Models     = []
            self.feopt          = np.empty([0], dtype=np.uint8)
            self.X_train_std    = X_train_std
            self.uncert_pr      = uncert_pr
            
            if N_consistancy == 0:
                self.Consistancy_Data = None
            else:    
                N_consistancy = int(N_consistancy / 2) * 2 + 1
                
                if uncert_pr:
                    self.Consistancy_Data = np.empty([0, N_consistancy, 2], dtype=float)
                else:
                    self.Consistancy_Data = np.empty([0, N_consistancy], dtype=float)

        
            if mode == 'classification':
                self.Mode          = 'classification'
                self.Classes       = classes
                self.Num_Classes   = len(classes)
                self.f_predictions = np.empty([0, self.Num_Classes], float)
                self.e_predictions = np.empty([0, self.Num_Classes], float)
                self.Outcomes      = np.empty([0], dtype=np.uint8)
                
            elif self.uncert_pr:
                self.Mode           = 'regression'
                self.f_predictions  = np.empty([0], dtype=float)
                self.e_predictions  = np.empty([0, 2], dtype=float)
                self.Outcomes       = np.empty([0], dtype=float)
                
            else:
                self.Mode           = 'regression'
                self.f_predictions  = np.empty([0], dtype=float)
                self.e_predictions  = np.empty([0], dtype=float)
                self.Outcomes       = np.empty([0], dtype=float)
                
        else:
            self.Read_From_File(file_handle)


           
    def Add_Sample(self, sample_scores, X_row, outcome, f_prediction, e_prediction, feopt, model, consistancy=None):
       
        if self.Mode == 'classification':
            self.f_predictions = np.vstack([self.f_predictions, np.asarray(f_prediction, dtype=float)])
            self.e_predictions = np.vstack([self.e_predictions, np.asarray(e_prediction, dtype=float)])
            self.Outcomes      = np.append(self.Outcomes, [int(outcome)])
        
        else:
            self.f_predictions = np.append(self.f_predictions, [f_prediction])
            
            if self.uncert_pr:
                self.e_predictions = np.vstack([self.e_predictions, np.asarray(e_prediction, dtype=float)])
            else:
                self.e_predictions = np.append(self.e_predictions, np.asarray(e_prediction, dtype=float))
            
            self.Outcomes      = np.append(self.Outcomes, [outcome])
            
            self.Max_Outcome = np.max(self.Outcomes)
            self.Min_Outcome = np.min(self.Outcomes)
        
        new_row    = np.array(sample_scores, dtype=float)
        scaled_row = np.array(sample_scores, dtype=float)

        max_score = np.max(abs(new_row))
        if max_score != 0.0:
            scaled_row = new_row / abs(max_score)
            
        self.Feature_Scores = np.vstack([self.Feature_Scores, new_row])
        self.Scaled_Scores  = np.vstack([self.Scaled_Scores,  scaled_row])
        self.Features       = np.vstack([self.Feature_Scores, np.array(X_row, dtype=float)])

        if feopt == None:
            self.feopt = None

        self.Exp_Models.append(model)

        if not isinstance(consistancy ,type(None)):
            self.Consistancy_Data = np.vstack([self.Consistancy_Data,  consistancy.reshape(1,-1,2)])
        
        self.Num_Samples += 1
        
        

    def Add_LIME_Sample(self, sample_scores, X_row, outcome, f_predictionn, e_prediction, model):
       
        if self.Mode == 'classification':
            self.f_predictions = np.vstack([self.f_predictions, np.asarray(f_prediction, dtype=float)])
            self.e_predictions = np.vstack([self.e_predictions, np.asarray(e_prediction, dtype=float)])
            self.Outcomes      = np.append(self.Outcomes, [int(outcome)])
        
        else:
            self.f_predictions = np.append(self.f_predictions, [f_prediction])
            self.Outcomes      = np.append(self.Outcomes, [outcome])

            e_prediction = np.asarray(e_prediction, dtype=float)
            #e_prediction = np.append(e_prediction, [0.0])#########################################################
            self.e_predictions = np.append([self.e_predictions, np.asarray(e_prediction, dtype=float)])

            
            self.Max_Outcome = np.max(self.Outcomes)
            self.Min_Outcome = np.min(self.Outcomes)
        
        new_row    = np.zeros([self.Num_Features], dtype=float)
        scaled_row = np.zeros([self.Num_Features], dtype=float)

        
        for item in sample_scores:
                     
            feature_index = self.Index_Of(item[0])
                
            new_row[feature_index] = item[1]     
            
            
        max_score = np.max(abs(new_row))                
        if max_score != 0.0:
            scaled_row = new_row / abs(max_score)
            
        self.Feature_Scores = np.vstack([self.Feature_Scores, new_row])
        self.Scaled_Scores  = np.vstack([self.Scaled_Scores,  scaled_row])
        self.Features       = np.vstack([self.Feature_Scores, np.array(X_row, dtype=float)])

        self.Exp_Models.append(model)

        self.Num_Samples += 1
        

    def Index_Of(self, feature_name):

        index = 0
        for name in self.Feature_Names:
            if name == feature_name:
                return index
            index += 1
                    

    def Copy_Rows (self, target):
        
        for row in range(self.Num_Samples):
            target.Add_Sample(sample_scores = self.Feature_Scores[row],
                              X_row         = self.Features[row],
                              outcome       = self.Outcomes[row],
                              f_prediction  = self.f_predictions[row],
                              e_prediction  = self.e_predictions[row],
                              feopt         = self.feopt,         
                              model         = self.Exp_Models[row])

      

    def Write_To_File(self, file_handle):

        pickle.dump(self.Feature_Names,    file_handle)
        pickle.dump(self.Num_Features,     file_handle)  
        pickle.dump(self.Num_Samples,      file_handle)  
        pickle.dump(self.Feature_Scores,   file_handle)
        pickle.dump(self.Scaled_Scores,    file_handle)
        pickle.dump(self.Mode,             file_handle)
        pickle.dump(self.uncert_pr,        file_handle)
        pickle.dump(self.Consistancy_Data, file_handle)

        if self.Mode == 'classification':
            pickle.dump(self.f_predictions, file_handle)
            pickle.dump(self.e_predictions, file_handle)
            pickle.dump(self.Outcomes,      file_handle)   
            pickle.dump(self.Classes,       file_handle)    
            pickle.dump(self.Num_Classes,   file_handle)
                
        else:
            pickle.dump(self.f_predictions, file_handle)
            pickle.dump(self.e_predictions, file_handle)
            pickle.dump(self.Outcomes,      file_handle)   
            pickle.dump(self.Min_Outcome,   file_handle)
            pickle.dump(self.Max_Outcome,   file_handle)


    def Read_From_File(self, file_handle):
              
        self.Feature_Names    = pickle.load(file_handle)  
        self.Num_Features     = pickle.load(file_handle)  
        self.Num_Samples      = pickle.load(file_handle)  
        self.Feature_Scores   = pickle.load(file_handle)
        self.Scaled_Scores    = pickle.load(file_handle)
        self.Mode             = pickle.load(file_handle)
        self.uncert_pr        = pickle.load(file_handle)
        self.Consistancy_Data = pickle.load(file_handle)

        if self.Mode == 'classification':
            self.f_predictions = pickle.load(file_handle)
            self.e_predictions = pickle.load(file_handle)
            self.Outcomes      = pickle.load(file_handle)   
            self.Classes       = pickle.load(file_handle)    
            self.Num_Classes   = pickle.load(file_handle)
                
        else:
            self.f_predictions = pickle.load(file_handle)
            self.e_predictions = pickle.load(file_handle)
            self.Outcomes      = pickle.load(file_handle)   
            self.Min_Outcome   = pickle.load(file_handle)
            self.Max_Outcome   = pickle.load(file_handle)

            
    def Feature_Counts(self, max_features=10, scaled=True, threshold=0.075):
        
        counts = np.zeros([self.Num_Features], dtype=np.int32)

        if scaled:
            for row in range(self.Num_Samples):
                for column in range(self.Num_Features):
                    if abs(self.Scaled_Scores[row][column]) > threshold:
                        counts[column] += 1
        else:     
            for row in range(self.Num_Samples):
                for column in range(self.Num_Features):
                    if abs(self.Feature_Scores[row][column]) > threshold:
                        counts[column] += 1
                        
        self.threshold  = threshold

        self.All_Counts = deepcopy(counts)
        
        if self.Num_Features <= max_features:
            
            self.Top_Scores   = self.Feature_Scores
            self.Top_Counts   = counts
            self.Top_Features = self.Feature_Names
            self.Top_Indicies  = np.arange(self.Num_Features, dtype=np.uint32)
            self.Max_Features = self.Num_Features

        else:

            
            
            top_indices = np.argpartition(counts, -max_features)[-max_features:]
            top_indices.sort()

            self.Top_Indicies = top_indices
            self.Max_Features = max_features
            
            self.Top_Counts   = counts[top_indices]
            self.Top_Scores   = self.Feature_Scores[:,top_indices]
            self.Top_Features = []
            
            for index in range(max_features):            
                self.Top_Features.append(self.Feature_Names[top_indices[index]])

            
              
    
    
    def Frequency_Plot(self, top_features=True, display_feature_list=False):

        fig, ax = plt.subplots()

        title = 'Feature Frequency of Explanations (above threshold ' + str(self.threshold) + \
                ') from ' + str(self.Num_Samples) + ' Samples for ' + self.Group_String()
       
        if top_features:
            if display_feature_list:
                ax.bar(x = np.arange(self.Max_Features), height = self.Top_Counts)
            else:
                ax.bar(x = self.Top_Features, height = self.Top_Counts)
        else:
            if display_feature_list:
                ax.bar(x = np.arange(self.Num_Features), height = self.All_Counts)
            else:
                ax.bar(x = self.Feature_Names, height = self.All_Counts)
        
        ax.set_ylabel('Feature Frequency')
        ax.set_ylim(ymin = 0, ymax = self.Num_Samples)
        ax.set_title(title)

        fig.tight_layout()
        plt.show()
        
        if display_feature_list:
            if top_features:
                self.Print_Top_Features(0)
            else:
                self.Print_Features(0)
                
                           
             
    @staticmethod        
    def Padding(num):
        if num < 10:    return '   '
        if num < 100:   return '  '
        if num < 1000:  return ' '
        if num < 10000: return ''
    
            
    def Print_Top_Features(self, index_1):
        for feature in range(self.Max_Features):
            print(feature+index_1, '- ', self.Padding(feature+1), self.Top_Features[feature])
            
            
    def Print_Features(self, index_1):
        for feature in range(self.Num_Features):
            print(feature+index_1, '- ', self.Padding(feature+1), self.Feature_Names[feature])
            
            
    def Violin_Plot(self, top_features=True, showextrema=True):

        fig, ax = plt.subplots()

        title = 'Violin Plot in Explanations from ' + str(self.Num_Samples) + ' Samples for ' + self.Group_String()

        if top_features:
            ax.violinplot(dataset = self.Top_Scores, vert=True, widths=0.5, showmeans=True, showextrema=showextrema)
        else:
            ax.violinplot(dataset = self.Feature_Scores, vert=True, widths=0.5, showmeans=True, showextrema=showextrema)
        
        ax.set_ylabel('Feature Explanation Scores')
        ax.set_title(title)
        plt.show()
        
        if top_features:
            self.Print_Top_Features(1)
        else:
            self.Print_Features(1)
            
        
            
    def Box_Plot(self, top_features=True, showfliers=False):

        fig, ax = plt.subplots()
        
        print('top_features',top_features)

        title = 'Box Plot in Explanations from ' + str(self.Num_Samples) + ' Samples for ' + self.Group_String()

        if top_features:
            print('top_features', self.Top_Scores.shape)
            ax.boxplot(x = self.Top_Scores, widths=0.5, \
                       patch_artist=True, showmeans=True, showfliers=showfliers)
        else:
            print('ALL features', self.Feature_Scores.shape)
            ax.boxplot(x = self.Feature_Scores, widths=0.5, \
                       patch_artist=True, showmeans=True, showfliers=showfliers)
        
        ax.set_ylabel('Feature Explanation Scores')
        ax.set_title(title)
        plt.show()
        
        if top_features:
            self.Print_Top_Features(1)
        else:
            self.Print_Features(1)
            
     

    def View_Explanation(self, instance, max_features=10):
            
        ###################################################################
        # Show Class Probabilities    
        if self.Mode == 'classification': 
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})
        
            f_predictions = self.f_predictions[instance,:].reshape(1, -1).ravel()
                        
            title1_info = str(instance) + ' (Outcome: ' + self.Classes[self.Outcomes[instance]] + ')'

            title1 = 'f_prediction Probabilities for Instance ' + title1_info


            bar_colorffs1  = []
            colour_index = 0
            for index in range(self.Num_Classes):
                bar_colorffs1.append(Feature_Statistics.colour_list[colour_index])
                
                colour_index += 1
                if colour_index == Feature_Statistics.colour_max:
                    colour_index = 0

            exp1 = ax1.barh(y = self.Classes, width = f_predictions, color=bar_colorffs1)
        
            ax1.bar_label(exp1, label_type='center', fmt='%.4f')

            ax1.set_xlabel('Class Probability')
            ax1.set_ylabel('Class Name')
            ax1.set_xlim(xmin = 0, xmax = 1)
            ax1.set_yticks(ticks=[])
       
            ax1.set_title(title1)
        

        ###################################################################
        ## for regression plot within the range of f_predictions the value
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]})
            
            f_prediction = self.f_predictions[instance]
            f_prediction = np.array([f_prediction])
 
            outcome_str = f"{self.Outcomes[instance]:.4f}"
            title1_info = str(instance) + ' (Outcome: ' + outcome_str + ')'
            
            title1 = 'Predicted value for Instance ' + title1_info


            if f_prediction[0] > 0:
                f_prediction_colour = 'tab:orange'
            else:
                f_prediction_colour = 'tab:blue'
            
            exp1 = ax1.barh(y = [''], width = f_prediction, color=f_prediction_colour)
        
            ax1.bar_label(exp1, label_type='center', fmt='%.4f')

            ax1.set_xlabel('Range of Predicted Values')

            x_min = np.min(self.f_predictions)
            x_max = np.max(self.f_predictions)
            ax1.set_xlim(xmin = x_min, xmax = x_max)

            ax1.set_title(title1)
        

        ###############################################################
        # Explanations    
    
        data_row = self.Feature_Scores[instance,:].reshape(1, -1).ravel()
        
        if max_features >= self.Num_Features:
            max_features  = self.Num_Features
            feature_names = self.Feature_Names
        else:
            # take a copy of the feature names as elements will be deleted for the graph
            feature_names = deepcopy(self.Feature_Names)

            # find the smallest elements in data_row and remove them, along with their feature name
            for remove in range(self.Num_Features - max_features):
                
                small       = 99.9
                small_index = 0
                for index in range(data_row.size):
                    if small > abs(data_row[index]):
                        small = abs(data_row[index])
                        small_index = index

                # remove smallest from list of names
                small_feature = feature_names[small_index]
                feature_names.remove(small_feature)
                
                # remove smallest from data row
                index = np.array([small_index])
                data_row = np.delete(data_row, index)
        
        title2 = 'Explanations from Instance ' + str(instance) 
        
        bar_colorffs2 = []
        for feature in range(max_features):
            if data_row[feature] > 0:
                bar_colorffs2.append('tab:orange')
            else:
                bar_colorffs2.append('tab:blue')

        exp2 = ax2.barh(y = feature_names, width = data_row, color=bar_colorffs2)
        
        ax2.axvline(0, color='lightgrey', linewidth=0.8)

        ax2.bar_label(exp2, label_type='center', fmt='%.4f')

        ax2.set_xlabel('Feature Effect')
        ax2.set_ylabel('Feature Name')
       
        ax2.set_title(title2)

        fig.tight_layout()
        plt.show()

    def Class_Fidelity_Graph(self):
               
        f_predictions = self.f_predictions[:,1]
        e_predictions = self.e_predictions[:,1]

        sorted_indices = np.argsort(f_predictions)

            
        fig, ax = plt.subplots()

        plt.scatter(x = np.arange(self.Num_Samples), y = f_predictions[sorted_indices], label = 'BB')
        plt.scatter(x = np.arange(self.Num_Samples), y = e_predictions[sorted_indices], label = 'Exp')
        plt.scatter(x = np.arange(self.Num_Samples), y = self.Outcomes[sorted_indices], label = 'Y Test')
               
        ax.set_xlabel('Sample')
        ax.set_ylabel('Model Prediction')
       
        ax.set_title('Model Predictions and Outcomes')
        
        ax.legend()
        plt.show()
        
        
    def Reg_Fidelity_Graph(self):
        
        fig, ax = plt.subplots()
            
        if self.feopt != None:
            plt.scatter(x = self.Outcomes, y = self.feopt, label = 'feopt',  marker='o')
            
        plt.scatter(x = self.Outcomes, y = self.f_predictions,      label = 'BB',  marker='x')
        
        if self.uncert_pr:
            plt.scatter(x = self.Outcomes, y = self.e_predictions[:,0], label = 'Exp', marker='+')
        else:
            plt.scatter(x = self.Outcomes, y = self.e_predictions,      label = 'Exp', marker='+')
               
        ax.set_xlabel('Y Test')
        ax.set_ylabel('Model Prediction')
       
        ax.set_title('Model Predictions vs Y Test')
        
        ax.legend()
        plt.show()
        
        
    def Fidelity(self):
        
        if self.Mode == 'classification':

            self.Class_Fidelity()      
                     
        else:
            self.Reg_Fidelity()
                     
            
    def Class_Fidelity(self):
        
        f_e_differences = np.square(self.f_predictions - self.e_predictions)

        f_outcomes = np.argmax(self.f_predictions, axis = 1)
        e_outcomes = np.argmax(self.e_predictions, axis = 1)

        f_score   = np.mean(self.Outcomes == f_outcomes)
        e_score   = np.mean(self.Outcomes == e_outcomes)
        f_e_score = np.mean(f_outcomes    == e_outcomes)
            
        print('BB(x) - exp(x) proba dif  Avg: ', np.mean(f_e_differences), ' var: ', np.var(f_e_differences),
              ' max: ', np.max(f_e_differences))
            
        print('Scores:')
        print('BB Model Score:  ', f_score)
        print('Exp Model Score: ', e_score)
        print('BB - Exp Score:  ', f_e_score)
            

            
    def Reg_Fidelity(self):
        
        if self.uncert_pr:
            exp_predictions = self.e_predictions[:,0]
        else:
            exp_predictions = self.e_predictions
            
        y_f_differences = np.abs(self.Outcomes - self.f_predictions )
            
        y_e_differences = np.abs(self.Outcomes - exp_predictions)
            
        f_e_differences = np.abs(self.f_predictions - exp_predictions)

        print('Average, Var, and Max Differences:')
            
        print('y - BB(x):          ', np.mean(y_f_differences), ' : ', np.var(y_f_differences),
              ' : ', np.max(y_f_differences))

        print('y - exp(x):         ', np.mean(y_e_differences), ' : ', np.var(y_e_differences),
              ' : ', np.max(y_e_differences))

        print('BB(x) - exp(x):     ', np.mean(f_e_differences), ' : ', np.var(f_e_differences),
              ' : ', np.max(f_e_differences))
        
        if self.uncert_pr:
            print('Average exp(x) var: ', np.mean(self.e_predictions[:,1]))
            
        self.fidelity = np.mean(y_e_differences) / np.mean(np.abs(self.Outcomes))
                

    def Jaccard_Values(self, top_k=5):

        jaccard_similarities = []
        jaccard_distances    = []
        
        evaluation_pairs = list(combinations(range(self.Num_Samples), 2))
                                
        for evaluation_pair in evaluation_pairs:
                                
            fs1 = self.Feature_Scores[evaluation_pair[0]]
            fs2 = self.Feature_Scores[evaluation_pair[1]]

            # Extracting indices of top-k features in both lists
            fs1 = set(np.argpartition(fs1, -top_k)[-top_k:])
            fs2 = set(np.argpartition(fs2, -top_k)[-top_k:])

            jaccard_similarity = len(fs1.intersection(fs2)) / len(fs1.union(fs2))
            jaccard_similarities.append(jaccard_similarity)                                
                                
            jaccard_distance = 1 - jaccard_similarity
            jaccard_distances.append(jaccard_distance)


        self.jaccard_similarities = np.mean(jaccard_similarities)
        
        print('Mean Jaccard Similarity: ', self.jaccard_similarities)
        print('Mean Jaccard Distance:   ', np.mean(jaccard_distances))

        
        
    def Group_String(self):
        return 'All Features'

    def Print_Data(self):
        print(self.Feature_Scores)
        
    def Print_Scaled(self):
        print(self.Scaled_Scores)

    def Number_Of_Samples(self):
        return self.Num_Samples
    
    def Get_Features(self):
        return self.Feature_Names
    
    def Get_Classes(self):
        return self.Classes
    
    def Data_Range(self):
        return self.Min_Outcome, self.Min_Outcome
        
    def Get_Ranges(self, num_ranges):
        
        outcome_range = self.Max_Outcome - self.Min_Outcome
        
        low_val   = self.Min_Outcome - (0.005 * outcome_range)
        increment = outcome_range * 1.01 / num_ranges
                
        return_ranges = np.empty([num_ranges + 1], dtype=float)

        for index in range(num_ranges + 1):
            return_ranges[index] = low_val + (index * increment)

        return return_ranges
    
    def add_Feature_Coeffs(self, Feature_Coeffs):
        
        self.Feature_Coeffs = Feature_Coeffs
        
        if self.Num_Samples > 0:        
            self.calculate_Feature_Coeffs()
            
    
    
    def calculate_Feature_Coeffs(self):
        
        mean_scores = np.mean(self.Feature_Scores, axis = 0)
        
        self.coeffs_ratio = self.Feature_Coeffs / mean_scores
        
        print('Mean Scores: ', mean_scores)
        print('Mean Coeffs: ', self.Feature_Coeffs)
        
    
    
    def delete_one(self):

        mean_scores = np.mean(self.Feature_Scores, axis = 0)
                              
        variance = np.zeros(self.Num_Features)
        
        for sample in range (self.Num_Samples):
            
            y = self.Exp_Models[sample].predict(self.Features[sample].reshape(1,-1))
        
            for feature in range(self.Num_Features):
            
                X_p = deepcopy(self.Features[sample])

                X_p[feature] = 0
            
                X_p = X_p.reshape([1,self.Num_Features])
                       
                y_p = self.Exp_Models[sample].predict(X_p)
           
                variance[feature] = variance[feature] + np.square(y - y_p)
            
        
        self.delete_one_var = variance / self.Num_Samples
        
        return self.delete_one_var
    
            
    def Compare_Models (self, model_b):
        
        model_diff = self.Feature_Scores - model_b.Feature_Scores
       
        self.model_diff_mean = np.mean(model_diff, axis = 0) / np.mean(model_diff)
        self.model_diff_std  = np.std(model_diff, axis = 0)  / np.mean(model_diff)
        
        print('Score Diff Mean: ', self.model_diff_mean)
        print('Score Diff SD:   ', self.model_diff_std)


    def Regression_Calibration (self, plot=True, title=''):

        # The Z scores in CRUDE are the difference in actual y and model prediction divided by the
        # uncertainty. In our case this is the difference in BB and explainer model, divided by the
        # uncertainty in the explainer model
        #
        Z_scores = (self.f_predictions - self.e_predictions[:,0]) / self.e_predictions[:,1]
        
        Z_scores = Z_scores.sort()
        Z_scores = Z_scores.reshape([-1,1])
        
        x_calib = (np.arange(self.Num_samples) + 1) / self.Num_samples
        x_calib = x_calib.reshape([-1,1])
        
        # Linear regress required for normalisation
        LR = LinearRegression(fit_intercept=False)
        LR.fit(x_calib, Z_scores)
        
        max_y = LR.predict(x_calib[-1,:])
        Z_scores = Z_scores / max_y

        self.calibration_MSE = 1 - mean_squared_error(x_calib, Z_scores)
        self.calibration_MAE = 1 - mean_absolute_error(x_calib, Z_scores)

        print('Calibration', self.calibration_MSE)
        
        if plot:
            
            fig, ax = plt.subplots()
            ax.scatter(x_calib, Z_scores)
            ax.plot([0,1],[0,1])
            plt.show()


    def Consistancy(self, std_bound, plot=True, title=''):       

        N_Points = np.size(self.Consistancy_Data, axis=1)
        print('N_Points: ',N_Points)
      
        mid_index = int(N_Points / 2)
        
        y_pert_all   = np.zeros(N_Points)
        y_uncert_all = np.zeros(N_Points)
        
        for sample in range(self.Num_Samples):
            
            if self.uncert_pr:
                y_mid    = self.Consistancy_Data[sample, mid_index, 0]           
                y_pert   = self.Consistancy_Data[sample, :, 0]
                y_uncert = self.Consistancy_Data[sample, :, 1]
                
                y_uncert_all = y_uncert_all + y_uncert / y_mid
            
            else:    
                y_mid  = self.Consistancy_Data[sample, mid_index]
                y_pert = self.Consistancy_Data[sample, :]
           
            y_pert_all = y_pert_all + ((y_pert - y_mid) / y_mid)
            
        y_pert_mean   = y_pert_all   / self.Num_Samples
        
        y_uncert_mean = y_uncert_all / self.Num_Samples
        
        if plot:
  
            fig, ax = plt.subplots()

            title = title + ' Consistancey_Plot'
       
            x = (np.arange(N_Points) - mid_index) / mid_index * std_bound
            #print('X: ', x, y_pert_mean)
            
            if self.uncert_pr:
                Add_Uncertainty_Plot(ax, x, y_pert_mean, y_uncert_mean)
            else:
                ax.plot(x, y_pert_mean)
        
            ax.set_xlabel('SD Perturbation')
            ax.set_ylabel('Normalised Prediction Change')
            ax.set_title(title)

            fig.tight_layout()
            plt.show()
            
            
####################################################################################################
    # Abstract class for containg groups of stats

class Group_Container(object):        
        
    def __init__(self, Reg_Stats):
        self.Stats_List = []

    def Feature_Counts(self, max_features=10, scaled=True, threshold=0.075):
        
        self.threshold = threshold
        
        for Element in self.Stats_List:
            Element.Feature_Counts(max_features, scaled, threshold)
            

    def Frequency_Plot(self, y_max=None, normalised=True, Overlay=False):

        if normalised:
            title = 'Normalised Feature Frequency of Explanations (above threshold ' + str(self.threshold) +')'
        else:
            title = 'Feature Frequency of Explanations (above threshold ' + str(self.threshold) +')'

        num_features   = len(self.Stats_List[0].Get_Features())
        summed_bottom = np.zeros([num_features])

        fig, ax = plt.subplots()

        for Element in self.Stats_List:
            if not Overlay:
                bottom = summed_bottom
            summed_bottom = Element.Plot_To_Axis(ax, normalised, bottom)
        
        if y_max == 'NoS':
            y_max = self.Num_Samples
            
        ax.set_ylabel('Feature Frequency')
        ax.set_ylim(ymin = 0, ymax = y_max)
        ax.set_title(title)

        fig.tight_layout()
        ax.legend()
        plt.show()
        
        self.Stats_List[0].Print_Features(0)
                
            
#############################################################################################
# Classes used to hold and contain statistics for ranges of regression outcomes


class Regression_Feature_Statistics(Feature_Statistics):

    def __init__(self, lower, upper, feature_names):
        
        super(Regression_Feature_Statistics, self).__init__(feature_names, 'regression')

        self.Lower = lower
        self.Upper = upper
        
        self.Label = f"{lower:.2f}" + '-' + f"{upper:.2f}"
         
        
    def Add_Sample(self, sample_scores, X_row, outcome, f_prediction, e_prediction, feopt, model):
        
        if outcome >= self.Lower and outcome <= self.Upper:
            Feature_Statistics.Add_Sample(self, sample_scores, X_row, outcome, f_prediction, e_prediction, feopt, model)
       
    def Add_LIME_Sample(self, sample_scores, X_row, outcome, f_prediction, e_prediction, model):
       
        if outcome >= self.Lower and outcome <= self.Upper:
            Feature_Statistics.Add_LIME_Sample(self, sample_scores, X_row, outcome, f_prediction, e_prediction, model)  
     
    def Plot_To_Axis(self, axis, normalised, bottom):

        if self.Num_Samples > 0:
            
            counts = self.All_Counts
        
            if normalised:
                counts = counts / self.Num_Samples
        
            axis.bar(x = np.arange(self.Num_Features), height = counts, \
                     label=self.Label, bottom = bottom)
        
            bottom = bottom + counts
        
        return bottom
    
            
    def Group_String(self):
        
        lower_str = f"{self.Lower:.3f}"
        upper_str = f"{self.Upper:.3f}"
        return 'Range:' + lower_str + '-' + upper_str


    
class Regression_Container(Group_Container):        
        
    def __init__(self, Reg_Stats, num_ranges):
        
        self.Num_Ranges  = num_ranges
        self.Num_Samples = Reg_Stats.Number_Of_Samples()
        
        outcome_range = Reg_Stats.Get_Ranges(num_ranges)

        self.Stats_List = []
        for index in range(num_ranges):
            
            new_element = Regression_Feature_Statistics(outcome_range[index], outcome_range[index+1], Reg_Stats.Get_Features())
            
            Reg_Stats.Copy_Rows(new_element)
                        
            self.Stats_List.append(new_element)
            
    def Element(self, index):
        return self.Stats_List[index]

            
#############################################################################################
# Classes used to hold and contain statistics for different class outcomes


class Class_Feature_Statistics(Feature_Statistics):

    def __init__(self, selected_class, feature_names, classes):
        
        super(Class_Feature_Statistics, self).__init__(feature_names, 'classification', classes)

        if isinstance(selected_class, int):
            self.Selected_Index = selected_class
            self.Selected_Class = classes[selected_class]
        else:
            self.Selected_Class = selected_class
            self.Selected_Index = classes.index(selected_class)
        
    def Add_Sample(self, sample_scores, X_row, outcome, f_prediction, e_prediction, feopt, model):
        
        if outcome == self.Selected_Index or outcome == self.Selected_Class:
            Feature_Statistics.Add_Sample(self, sample_scores, X_row, outcome, f_prediction, e_prediction, feopt, model)
       
    def Add_LIME_Sample(self, sample_scores, X_row, outcome, f_prediction, e_prediction, model):
       
        if outcome == self.Selected_Index or outcome == self.Selected_Class:
            Feature_Statistics.Add_LIME_Sample(self, sample_scores, X_row, outcome, f_prediction, e_prediction, model)  
     
    def Plot_To_Axis(self, ax, normalised, bottom):
    
        counts = self.All_Counts
        
        if normalised and self.Num_Samples > 0:
            counts = counts / self.Num_Samples
        
        ax.bar(x = np.arange(self.Num_Features), height = counts, \
               label=self.Selected_Class, bottom = bottom)
        
        bottom = bottom + counts
        
        return bottom

            
    def Group_String(self):
        return ' Class ' + self.Selected_Class

                
class Classes_Container(Group_Container):        
        
    def __init__(self, Class_Stats):
        
        self.Classes     = Class_Stats.Get_Classes()
        self.Num_Samples = Class_Stats.Number_Of_Samples()
        
        self.Stats_List = []
        for index in self.Classes:
            
            new_element = Class_Feature_Statistics(index, Class_Stats.Get_Features(), self.Classes)
            
            Class_Stats.Copy_Rows(new_element)
                        
            self.Stats_List.append(new_element)
            
            
    def Element(self, index):
              
        if isinstance(index, int):
            return self.Stats_List[index]
        else:
            return self.Stats_List[self.Classes.index(index)]
        
    
        
