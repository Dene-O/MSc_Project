import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

#from sklearn.utils import check_random_state


class Feature_Statistics(object):

    #list of colours and its range for dipslaying graphs
    colour_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:yellow']
    colour_max  = len(colour_list)
    
                       
    def __init__(self, feature_names, mode='classification', classes=None):
        
        self.Feature_Names  = feature_names
        self.Num_Features   = len(feature_names)
        self.Num_Samples    = 0
        self.Feature_Scores = np.empty([0, self.Num_Features], dtype=float)
        self.Scaled_Scores  = np.empty([0, self.Num_Features], dtype=float)
        
        if mode == 'classification':
            self.Mode        = 'classification'
            self.Classes     = classes
            self.Num_Classes = len(classes)
            self.Predictions = np.empty([0, self.Num_Classes], float)
            self.Outcomes    = np.empty([0], dtype=np.uint8)
                
        else:
            self.Mode = 'regression'
            self.Predictions    = np.empty([0], dtype=float)
            self.Outcomes       = np.empty([0], dtype=float)
        
    
    def Add_Sample(self, sample, outcome, prediction):
       
        if self.Mode == 'classification':
            self.Predictions = np.vstack([self.Predictions, np.asarray(prediction, dtype=float)])
            self.Outcomes    = np.append(self.Outcomes, [int(outcome)])
        
        else:
            self.Predictions = np.append(self.Predictions, [prediction])
            self.Outcomes    = np.append(self.Outcomes, [outcome])
            
            self.Max_Outcome = np.max(self.Outcomes)
            self.Min_Outcome = np.min(self.Outcomes)
        
        new_row    = np.array(sample, dtype=float)
        scaled_row = np.array(sample, dtype=float)

        max_score = 0.0  
        for feature_value in sample:                                 
            if max_score < abs(feature_value):
                max_score = abs(feature_value)
                
        if max_score != 0.0:
            scaled_row = new_row / abs(max_score)
            
        self.Feature_Scores = np.vstack([self.Feature_Scores, new_row])
        self.Scaled_Scores  = np.vstack([self.Scaled_Scores,  scaled_row])

        self.Num_Samples += 1
        
        

    def Add_LIME_Sample(self, sample, outcome, prediction):
       
        if self.Mode == 'classification':
            self.Predictions = np.vstack([self.Predictions, np.asarray(prediction, dtype=float)])
            self.Outcomes    = np.append(self.Outcomes, [int(outcome)])
        
        else:
            self.Predictions = np.append(self.Predictions, [prediction])
            self.Outcomes    = np.append(self.Outcomes, [outcome])
        
            self.Max_Outcome = np.max(self.Outcomes)
            self.Min_Outcome = np.min(self.Outcomes)
        

        new_row    = np.zeros([self.Num_Features], dtype=float)
        scaled_row = np.zeros([self.Num_Features], dtype=float)

        max_score = 0
        
        for item in sample:
                     
            feature_index = self.Index_Of(item[0])
                
            new_row[feature_index] = item[1]     
            
            max_score += abs(item[1])
                
        if max_score != 0.0:
            scaled_row = new_row / abs(max_score)
            
        self.Feature_Scores = np.vstack([self.Feature_Scores, new_row])
        self.Scaled_Scores  = np.vstack([self.Scaled_Scores,  scaled_row])

        self.Num_Samples += 1
        

    def Index_Of(self, feature_name):

        index = 0
        for name in self.Feature_Names:
            if name == feature_name:
                return index
            index += 1

            
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
                        
        self.threshold    = threshold

        self.All_Counts = deepcopy(counts)
        
        if self.Num_Features <= max_features:
            
            self.Top_Scores   = self.Feature_Scores
            self.Top_Counts   = counts
            self.Top_Features = self.Feature_Names
            self.Top_Indicies  = np.arange(self.Num_Features, dtype=np.uint32)
            self.Max_Features = self.Num_Features

        else:

            # copy the values in counts for later
            original_counts = deepcopy(counts)
            
            # determine the features and their indices with the highest counts
            top_indices = np.empty([max_features], dtype=np.uint32)
            for outer in range(max_features):
                
                max_index =  0
                max_count = -1
                for inner in range(self.Num_Features):
                    if counts[inner] > max_count:
                        max_count = counts[inner]
                        max_index = inner

                # store index of highest count in top_indices, but remove that count for the next loop
                top_indices[outer] = max_index
                counts[max_index]  = -1
            
 
            # the feature indices are in order of the highest feature counts, sort them in indices order
            top_indices.sort()
                        
            #assign
            self.Top_Scores = np.empty([self.Num_Samples, max_features], dtype=float)
            self.Top_Counts = np.empty([max_features], dtype=np.int32)
            self.Top_Features = []
            for index in range(max_features):
                
                self.Top_Counts[index] = original_counts[top_indices[index]]
                
                self.Top_Features.append(self.Feature_Names[top_indices[index]])
                
                self.Top_Scores[:,index] = self.Feature_Scores[:,top_indices[index]]
                
                
            self.Top_Indicies = top_indices
            self.Max_Features = max_features
            
    
    
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
                self.Print_Top_Features()
            else:
                self.Print_Features()
                
                
    @staticmethod        
    def Padding(num):
        if num < 10:    return '   '
        if num < 100:   return '  '
        if num < 1000:  return ' '
        if num < 10000: return ''
    
            
    def Print_Top_Features(self):
        for feature in range(self.Max_Features):
            print(feature+1, '- ', self.Padding(feature+1), self.Top_Features[feature])
            
            
    def Print_Features(self):
        for feature in range(self.Num_Features):
            print(feature+1, '- ', self.Padding(feature+1), self.Feature_Names[feature])
            
            
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
            self.Print_Top_Features()
        else:
            self.Print_Features()
            
        
            
    def Box_Plot(self, top_features=True, showfliers=True):

        fig, ax = plt.subplots()

        title = 'Box Plot in Explanations from ' + str(self.Num_Samples) + ' Samples for ' + self.Group_String()

        if top_features:
            ax.boxplot(x = self.Top_Scores, widths=0.5, \
                       patch_artist=True, showmeans=True, showfliers=showfliers)
        else:
            ax.boxplot(x = self.Feature_Scores, widths=0.5, \
                       patch_artist=True, showmeans=True, showfliers=showfliers)
        
        ax.set_ylabel('Feature Explanation Scores')
        ax.set_title(title)
        plt.show()
        
        if top_features:
            self.Print_Top_Features()
        else:
            self.Print_Features()
            
                
    

    def View_Explanation(self, instance, max_features=10):
            
        ###################################################################
        # Show Class Probabilities    
        if self.Mode == 'classification': 
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 2]})
        
            predictions = self.Predictions[instance,:].reshape(1, -1).ravel()
                        
            title1_info = str(instance) + ' (Outcome: ' + self.Classes[self.Outcomes[instance]] + ')'

            title1 = 'Prediction Probabilities for Instance ' + title1_info


            bar_colors1  = []
            colour_index = 0
            for index in range(self.Num_Classes):
                bar_colors1.append(Feature_Statistics.colour_list[colour_index])
                
                colour_index += 1
                if colour_index == Feature_Statistics.colour_max:
                    colour_index = 0

            exp1 = ax1.barh(y = self.Classes, width = predictions, color=bar_colors1)
        
            ax1.bar_label(exp1, label_type='center', fmt='%.4f')

            ax1.set_xlabel('Class Probability')
            ax1.set_ylabel('Class Name')
            ax1.set_xlim(xmin = 0, xmax = 1)
       
            ax1.set_title(title1)
        

        ###################################################################
        ## for regression plot within the range of predictions the value
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 4]})
            
            prediction = self.Predictions[instance]
            prediction = np.array([prediction])
 
            outcome_str = f"{self.Outcomes[instance]:.4f}"
            title1_info = str(instance) + ' (Outcome: ' + outcome_str + ')'
            
            title1 = 'Predicted value for Instance ' + title1_info


            if prediction[0] > 0:
                prediction_colour = 'tab:orange'
            else:
                prediction_colour = 'tab:blue'
            
            exp1 = ax1.barh(y = [''], width = prediction, color=prediction_colour)
        
            ax1.bar_label(exp1, label_type='center', fmt='%.4f')

            ax1.set_xlabel('Range of Predicted Values')

            x_min = np.min(self.Predictions)
            x_max = np.max(self.Predictions)
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
        
        bar_colors2 = []
        for feature in range(max_features):
            if data_row[feature] > 0:
                bar_colors2.append('tab:orange')
            else:
                bar_colors2.append('tab:blue')

        exp2 = ax2.barh(y = feature_names, width = data_row, color=bar_colors2)
        
        ax2.axvline(0, color='lightgrey', linewidth=0.8)

        ax2.bar_label(exp2, label_type='center', fmt='%.4f')

        ax2.set_xlabel('Feature Effect')
        ax2.set_ylabel('Feature Name')
       
        ax2.set_title(title2)

        fig.tight_layout()
        plt.show()
        
              
    def Group_String(self):
        return 'All Samples'

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
        
    def Copy_Rows (self, target):
        
        for row in range(self.Num_Samples):
            target.Add_Sample(self.Feature_Scores[row], self.Outcomes[row], self.Predictions[row])


    def Get_Ranges(self, num_ranges):
        
        outcome_range = self.Max_Outcome - self.Min_Outcome
        
        low_val   = self.Min_Outcome - (0.005 * outcome_range)
        increment = outcome_range * 1.01 / num_ranges
                
        return_ranges = np.empty([num_ranges + 1], dtype=float)

        for index in range(num_ranges + 1):
            return_ranges[index] = low_val + (index * increment)

        return return_ranges
            
#############################################################################################
# Classes used to hold and contain statistics for ranges of regression outcomes

class Regression_Feature_Statistics(Feature_Statistics):

    def __init__(self, lower, upper, feature_names):
        
        super(Regression_Feature_Statistics, self).__init__(feature_names, 'regression')

        self.Lower = lower
        self.Upper = upper
        
    def Add_Sample(self, sample, outcome, prediction):
        
        if outcome >= self.Lower and outcome <= self.Upper:
            Feature_Statistics.Add_Sample(self, sample, outcome, prediction)
       
    def Add_LIME_Sample(self, sample, outcome, prediction):
       
        if outcome >= self.Lower and outcome <= self.Upper:
            Feature_Statistics.Add_LIME_Sample(self, sample, outcome, prediction)  
     
    def Group_String(self):
        
        lower_str = f"{self.Lower:.3f}"
        upper_str = f"{self.Upper:.3f}"
        return 'Range:' + lower_str + '-' + upper_str


    
class Regression_Container(object):        
        
    def __init__(self, Reg_Stats, num_ranges):
        
        self.Num_Ranges = num_ranges
        
        outcome_range = Reg_Stats.Get_Ranges(num_ranges)

        self.Stats_List = []
        for index in range(num_ranges):
            
            new_element = Regression_Feature_Statistics(outcome_range[index], outcome_range[index+1], Reg_Stats.Get_Features())
            
            Reg_Stats.Copy_Rows(new_element)
                        
            self.Stats_List.append(new_element)
            
    def Feature_Counts(self, max_features=10, scaled=True, threshold=0.075):
        
        for Element in self.Stats_List:
            Element.Feature_Counts(max_features, scaled, threshold)
            
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
        
    def Add_Sample(self, sample, outcome, prediction):
        
        if outcome == self.Selected_Index or outcome == self.Selected_Class:
            Feature_Statistics.Add_Sample(self, sample, outcome, prediction)
       
    def Add_LIME_Sample(self, sample, outcome, prediction):
       
        if outcome == self.Selected_Index or outcome == self.Selected_Class:
            Feature_Statistics.Add_LIME_Sample(self, sample, outcome, prediction)  
     
    def Group_String(self):
        return self.Selected_Class + ' Class'

                
class Classes_Container(object):        
        
    def __init__(self, Class_Stats):
        
        self.Classes = Class_Stats.Get_Classes()
        
        self.Stats_List = []
        for index in self.Classes:
            
            new_element = Class_Feature_Statistics(index, Class_Stats.Get_Features(), self.Classes)
            
            Class_Stats.Copy_Rows(new_element)
                        
            self.Stats_List.append(new_element)
            
    def Feature_Counts(self, max_features=10, scaled=True, threshold=0.075):
        
        for Element in self.Stats_List:
            Element.Feature_Counts(max_features, scaled, threshold)
            
    def Element(self, index):
              
        if isinstance(index, int):
            return self.Stats_List[index]
        else:
            return self.Stats_List[self.Classes.index(index)]
        
    
        
