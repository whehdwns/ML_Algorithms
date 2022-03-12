#!/usr/bin/env python
# coding: utf-8
#Programing_Project_3_DCho.ipynb

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import os
import glob
from collections import Counter # Mode
import warnings #Remove Warning Message
warnings.filterwarnings('ignore')

path = os.getcwd()
all_files = glob.glob(path + "/dataset/*.data")
filesnames = os.listdir('dataset/')

dataset_header = [
                [   "sex", "length", "diameter", "height", "whole_height",
                    "shucked_height", "viscera_weight", "shell_weight", "rings" ], 
                [   "sample_code_number", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
                    "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin",
                    "normal_nucleoli", "mitosis", "class"],
                [   "buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
                [   "X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"],
                [   "class", "infants", "water", "budget", "physician", "salvador", "religious", "satellite",
                    "aid", "missile", "immigration", "synfuels", "education", "superfund", "crime", 
                    "duty_free_exports", "eaa_rsa"],
                [   "vendor", "model", "myct", "mmin", "mmax", "cach", "chmin", "chmax", "PRP", "ERP"] 
]


# ## Loading Data
# Load Data
# 1. It loads dataset from the folder. 
# 2. It checks whether the original data already has header or not. 
# 3. If there is no header in the data, it reads .data file as csv with header.
# 4. If there is header in the data, it reads .data file as csv. 
def open_csv_dataset(dataset_keyword, column_header):
    for i in all_files:
        if dataset_keyword in i:
            indexed = all_files.index(i)
            if column_header == True:
                df = pd.read_csv(i, header=None, names = dataset_header[indexed])
            else:
                df = pd.read_csv(i)
    return df


# ## Handling Missing Values
# Handling Missing Values
# 1. It replaces "?" values to NaN value in the dataframe.
# 2. If there is no null values in the dataframe, it returns dataframe
# 3. If there is null values in the dataframe, it fills missing values with the feature (column) mean.
def handling_missing_values(df):
    df = df.replace('?', np.NaN)
    if df.isnull().values.any() == False:
        return df
    else:
        for i in range(0, len(df.isnull().sum().values)):
            if df.isnull().sum().values[i] >0:
                missing_data = df.isnull().sum().index[i]
        df[missing_data]= pd.to_numeric(df[missing_data])
        df[missing_data]= df[missing_data].fillna(df[missing_data].mean())
    return df


# ## Handling Categorical Data
# Unique value in columns
# 1. It categorizes all columns in the dataset. 
def view_unique_value_in_columns(df):
    for i in df.columns.values:
        print(categorize_dataset(df, i))

# Unique value in single columns
# 1. It categorizes single columns in the dataset. 
def view_unique_value_in_single_column(df, columns):
    print(categorize_dataset(df, columns))


# Categorize Dataset
# 1. To categroize dataset, it uses unique function.
# 2. For unique function, it returns in order of appearance.
# 3. It shows columns and numeric values in each feature values. 
def categorize_dataset(df, columns):
    dicts = {}
    unique_list = list(df[columns].unique())
    for i in range(0, len(unique_list)):
        dicts[unique_list[i]] = i
    return {columns: dicts}


# Replace to numeric single/multiple columns in the dataset
# 1. Based on categorize dataset function, it replaces string value to numeric values for single/multiple columns
def replacing_string_to_numeric_multiple_columns(df, string_columns ):
    for i in string_columns:
        df = df.replace(categorize_dataset(df,i))
    return df

# Replace to numeric all columns in the dataset
# 1. Based on categorize dataset function, it replaces string value to numeric values for all columns.
def replacing_string_to_numeric_all_columns(df):
    for i in df.columns.values:
        df = df.replace(categorize_dataset(df, i))
    return df


# ## Categorical Distribution for Computer Hardware Dataset
# Compter Categorical Distribution
# It categorizes the machine data based on names file.
def computer_categorical_distribution(x):
    if 0<=x <=20:
        return 0
    if 21<=x <=100:
        return 1
    if 101<=x<=200:
        return 2
    if 201 <=x<=300:
        return 3
    if 301 <= x<=400:
        return 4
    if 401<=x<=500:
        return 5
    if 501<=x<=600:
        return 6
    if x >600:
        return 7


# ## Log Transform
# Log Transform
# 1. This function is used to apply for Forest Fires data. 
# 2. Based on note, it shows the output area is very skewed toward 0.0. The authors recommend a log transform.
# 3. It log transform certain columns in the dataset. 
def log_transform(x):
    return np.log(x + 1)


# ## Discretization
# Discretization
# This function is used to transform real-valued data into a series of discretized values
# 1. For discretization, it has two method: Equal Width , Equal Frequency
# 2. For Equal width, it divides the data into n intervals of equal size.
#  2.1 Width of the k interval is (max - min) /n. 
#  2.2 It replaces discretized values with dataset
# 3. For Equal Frequency, it uses pandas function: qcut().
#  3.1 Based on padas, it discretize variable into equal-sized buckets based on rank or based on sample quantiles.
#  3.2 It replaces discretized values with the dataset
def discretization(df, n, method, columns):
    if method =="equal_width":
        for i in columns:
            max_val = np.amax(df[i].values)
            min_val = np.amin(df[i].values)
            bin_size = (max_val - min_val) / n
            result =[]
            for j in df[i].values:
                bin_num = int( j // bin_size )
                if bin_num > n - 1:
                    bin_num = n - 1
                result.append( bin_num )
            df[i] = df[i].replace(df[i].values,result)
            result.clear()
        return df
    if method == "equal_frequency":
        for i in columns:
            result = pd.qcut(df[i].values,n, labels=False, duplicates='drop')
            df[i] = df[i].replace(df[i].values,result)
        return df


# ## Standardization
# Split dataset
# 1. It splits the dataset into two: training set and test set.
# 2. Training set has 80% of original dataset. 
# 3. Testing set has 20% of original dataset. 
def split_dataset(df, train_perc):
    train_end_ind = int(round(df.shape[0] * train_perc))
    train = df.iloc[0:train_end_ind]
    test = df.iloc[train_end_ind:-1]
    return (train, test)

# Z-score standardization
# It computes z-score by (observed value - mean of the sample) / standard deviation of the sample
def z_score_standardization(df):
    z_score = (df-df.mean())/df.std()
    return z_score

# Standardization
# 1. It applies z-score standardization for training set and testign set. 
def Standardization(training, testing):
    training_zscore = z_score_standardization(training)
    testing_zscore = z_score_standardization(testing)
    return (training_zscore, testing_zscore)


# ## Cross-validation
#K-fold cross validation
# For this k-fold cross valiation, it applied startification.
# 1. With training set, it splits training set into k-equaled size. 
# 2. It returns k-equal-sized training partitions. 
def cross_validation(df, k):
    df_size= len(df)
    df_size = df_size//k 
    remainder = df_size %k
    df_folds = []
    start = 0
    for i in range(0,k):
        if i < remainder:
            fold =  df.iloc[start : start+df_size+1]
            df_folds.append(fold)
            start +=  df_size + 1
        else:
            fold =  df.iloc[start : start+df_size]
            df_folds.append(fold)
            start +=  df_size
    return df_folds

# Cross Validation Regression
# 1. Unlike Cross validation classification, it samples uniformly across all of the response values.
# 2. It sorts the data by predictor. Then, It takes fifth point for a given fold.
# 3. It returns k-equal-sized training partitions. 
def cross_validation_regression(df, k, var):
    df = df.sort_values(var, axis=0)
    df_size= len(df)
    df_size = df_size//k 
    remainder = df_size %k
    df_folds = []
    start = 0
    for i in range(0,k):
        if i < remainder:
            fold = df.iloc[::5,].reset_index(drop=True)
            fold =  df.iloc[start : start+df_size+1]
            df_folds.append(fold)
            start +=  df_size + 1
        else:
            fold = df.iloc[::5,].reset_index(drop=True)
            fold =  df.iloc[start : start+df_size]
            df_folds.append(fold)
            start +=  df_size
    return df_folds


# ## Evaluation Metrics

# Evaluation Metrics
# It used to evaluate the efficacy of a machine learning algorithm on a dataset. 
# 1. Classification score
#  1.1 It computes accuracy score between predicted values and observed values
# 2. MSE
#  2.1 It compute MSE by sum of square of difference between actual and predicted and divide by number of points.
# 3. MAE
#  3.1 It compute MAE by sum of absolute of (prediction - true value) and divide by number of points.
# 4. R Square
#  4.1 R squared is computed by 1 - RSS (sum of squares of residuals)/ TSS (total sum of squares)
# 5. Pearson’s correlation
#  5.1 Pearson's correlation is computed by Covariance of X and Y / (stadard deviation of X * stadard deviation of Y)
def evaluation_metrics(y_true, y_pred, method):
    diff = np.subtract(y_true, y_pred)
    if method =="classification score":
        count = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_true[i]:
                count +=1
        accuracy_score = count / len(y_pred)
        return accuracy_score
    if method =="MSE":
        return np.mean(diff**2)
    if method =="MAE":
        return np.mean(abs(diff))
    if method =="R square":
        y_bar = y.mean()
        TSS = ((y-y_bar)**2).sum()
        RSS = (diff**2).sum()
        return 1 - (RSS/TSS)
    if method =="Pearson_correlation":
        covariance = np.cov(y_true, y_pred)
        pearson = covariance / (np.std(y_true) * np.std(y_pred))
        return pearson


# ## ID3
# ID3 Node to construct the decision tree
#    target: predictor class assigned to node
#    parent: parent node for the attribute
#    parent_feature: parent attribute value
#    attribute: attribute node
#    attribute_feature: attribute values
#    children: All the data for the certain child
#    child: Child node of the parents
#    prune_tag: indicator whether we prune the node or not.
class ID3_Node:
    def __init__(self, target):
        self.target = target
        self.parent = None
        self.parent_feature = []
        self.attribute = None
        self.attribute_feature = []
        self.children= []
        self.child = {}
        self.prune_tag = False

#Entropy
# It computes the amount of information contain within the target. 
# It can be computed by -(fraction)*log(fraction).
def entropy(target):
    elements, counts = np.unique(target, return_counts=True)
    entropy_node = 0
    for i in range(len(elements)):
        fraction = counts[i] / np.sum(counts)
        entropy_node += (-fraction)*(np.log2(fraction))
    return entropy_node


# ### Categorical

#Compute_gain
# It comptues the Information gain for each attribute.
# For information gain, it can be comptued by subtracting target entropy to estimated entropy.
def compute_gain(df, column_attr, target):
    target_entropy = entropy(df[target])
    elements, counts = np.unique(df[column_attr], return_counts=True)
    total_weight_entropy =[]
    for i in range(len(elements)):
        fraction = counts[i] / np.sum(counts)
        entropy_node = entropy(df[df[column_attr]==elements[i]][target])
        weight_entropy = fraction*entropy_node
        total_weight_entropy.append(weight_entropy)
    expected_entropy = np.sum(total_weight_entropy)
    gain = target_entropy - expected_entropy
    return gain


# Find Highest Gain
# ID3 uses information gain for attribute selection in decision tree. 
# It computes the information gain for each attribute using compute_gain function.
# Then, it returns the attributes that has highest information gain. 
def find_highest_gain(df, target):
    highest_gain_list = []
    columns_feature = df.columns[df.columns!=target]
    for i in columns_feature:
        highest_gain_list.append(compute_gain(df, i, target))
    highest_gain = columns_feature[highest_gain_list.index(max(highest_gain_list))]
    highest_gain_ratio = max(highest_gain_list)
    return highest_gain, highest_gain_ratio


# ID3_categorical
#  It handles the categorical attributes of the dataset. 
#  It initializes the decision tree by setting root node as the attribute with highest information gain.
#  It gets the attribute feature value for the root node.
#  It splits the instance, and continues to build tree. 
# Stopping Criteria
#  If every instance has same class label, it returns most unique class
#  If there is no attribute to split, it returns attribute with highest information gain.
def ID3_categorical(df, predictor, features, most_common_label=None):
    if len(np.unique(df[predictor])) <=1:
        most_common_label = np.unique(df[predictor])[0]
        return ID3_Node(most_common_label)
    elif len(features) ==0:
        return most_common_label
    else:
        most_common_label = np.unique(df[predictor])[np.argmax(np.unique(df[predictor], return_counts=True)[1])]
        TREE = ID3_Node(most_common_label)
        attribute_selected = find_highest_gain(df, predictor)[0]
        TREE.attribute = attribute_selected

        best_feature = list(df[find_highest_gain(df, predictor)[0]].unique())
        TREE.attribute_feature = best_feature
        
        features = [i for i in features if i != attribute_selected]
        for i in TREE.attribute_feature:
            sub_data = df[df[attribute_selected] == i]
            subtree = ID3_categorical(sub_data, predictor, features, most_common_label)
            subtree.children = sub_data
            subtree.parent =  attribute_selected
            subtree.parent_feature = i
            
            TREE.child[i] = subtree
        return TREE

# Predict ID3
#  If the node is left node, it return class label. 
#  If the node does't have to prune, it keeps predicts until leaf node for every child node.
#  If the node have to prune, then it returns common unique class for the subtree. 
def predict_ID3(TREE, row_attr, predictor):
    if len(TREE.child) == 0:
        return TREE.target
    else:
        feature_value = row_attr[TREE.attribute]
        if feature_value in TREE.child and TREE.child[feature_value].prune_tag == False:
            return predict_ID3(TREE.child[feature_value], row_attr, predictor)
        else:
            pred_y = []
            for i in TREE.attribute_feature:
                pred_y.append(np.unique(TREE.child[i].children[predictor])[0])
            return np.unique(pred_y)[0]


# ### Numerical
# Select_best_numeric
#  This function is to select best attribute and gain ratio for the numeric dataset. 
#  For each of the row in the dataset, it determins whether the row is cancidate split points.
#  It decides where to split where the class changes. 
#  It splits the data into two branches(left, right).
#  It computes information gain, and best gain ratio for the split.
#  It returns the best split points and best gain ratio.
def select_best_numeric(df, predictor, attribute):
    binary_split= []
    attribute_gain = []
    best_gain_ratio = 0
    best_f = None
    total = df.shape[0]
    pre_entropy = entropy(df[predictor])
    for i in range(1, len(df)):
        if df.loc[i, predictor] != df.loc[i + 1, predictor]:
            middle_point = (df.loc[i, attribute] + df.loc[i+1, attribute]) / 2
            left = df.loc[df[attribute] < middle_point]
            right = df.loc[df[attribute] > middle_point]
            entropy_left = (len(left)/total)* entropy(left[predictor])
            entropy_right =(len(right)/total)* entropy(right[predictor])
            total_entropy = entropy_left + entropy_right
            info_gain = pre_entropy - total_entropy
            attribute_gain.append(info_gain)
            binary_split.append(middle_point)
    best_gain_ratio = max(attribute_gain)
    best_split = binary_split[attribute_gain.index(max(attribute_gain))]
    return best_split, best_gain_ratio, attribute                      


# Best Attribute selection
# It iterates through all attribute except target feature
# It sorts the data on the attribute and target feature.
# Using Select_best_numeric function, it computes the information for each attribute.
# It finds the attribute that has highest gain.
# It returns the attributes that has highest information gain, and split points of the attribute. 
def best_attribute_selection(df, predictor):
    info_gain = []
    for i in df.columns[df.columns!=predictor]:
        new_df = df.sort_values(by =[i, predictor])
        info_gain.append(select_best_numeric(new_df, predictor, i)[1])
    if select_best_numeric(df, predictor, i)[1] == max(info_gain):
        selected_attribute = (select_best_numeric(df, predictor, i)[2], select_best_numeric(df, predictor, i)[0])
    return selected_attribute


# ### ID3 Score
# ID3_score
# It determines what type of attribute we are handlin either categorical or numerical.
# It predicts the class label for he dataset, and evaluate the performance using classification score. 
# It returns the accuracy score. 
# * For this project, I was able to figure out how to get best numerical attribute, 
#   but I wasn't able to implement the decision tree with numerical attribute. 
def ID3_score(TREE, df, attr_type, predictor):
    pred_list = []
    for i in range(len(df)):
        if attr_type == 'categorical':
            prediction = predict_ID3(TREE, df.iloc[i, :], predictor)
        elif attr_type == 'numerical':
            return print("best attribute" + str(best_attribute_selection(df, predictor)))
        pred_list.append(prediction)
    score = evaluation_metrics(df[predictor].values, pred_list, "classification score")
    return score


# ### Pruning (Reduced Error Pruning)
# Pruning
# It uses the tree that already grown to completion. 
# By iterating all the child node in the decision tree, it tagged the node for prune.
# If the tagged tree performs better than untagged tree, then tagged tree becomes new tree.
# It continues to pruning until no improvement occurs or if there is no node left to test. 
# It returns the tree with prunned tag tree.
def pruning(TREE, TREE_v1, tune_df, attr_type, predictor):
    branches = TREE_v1.child
    if len(branches) == 0:
        privous_score = ID3_score(TREE, tune_df, attr_type, predictor)
        TREE_v1.prune_tag = True
        new_score = ID3_score(TREE, tune_df, attr_type, predictor)
        if privous_score > new_score:
            TREE_v1.prune_tag = False
        return TREE
    else:
        for value, subtree in branches.items():
            pruned_tree = pruning(TREE, subtree, tune_df, attr_type, predictor)
        privous_score = ID3_score(pruned_tree, tune_df, attr_type, predictor)
        TREE_v1.prune_tag = True
        new_score = ID3_score(pruned_tree, tune_df, attr_type, predictor)
        if privous_score > new_score:
            TREE_v1.prune_tag = False
        return pruned_tree


# ### ID3 Evaluation
# ID3
#  It computes ID3 without pruned and with pruned. 
#  It evaluates the performance using ID3_Score function.
#  It returns the perofrmance of each of the fold. 
#  For reduced error pruning, it uses 20% of dataset to do pruning process.
#  Then, test the pruned tree with 80% of cross-validation. 
def ID3(df, predictor, test, attr_type, pruned:bool):
    score_list = []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        testing_val_y = df[i][predictor].values
        columns_feature = k_validate.columns[k_validate.columns!=predictor]
        if attr_type == 'categorical':
            if pruned == False:
                Tree = ID3_categorical(k_validate, predictor, columns_feature)
                Score = ID3_score(Tree, df[i], attr_type, predictor)
                Score = round((Score*100),2)
                score_list.append(Score)
            else:
                Tree = ID3_categorical(k_validate, predictor, columns_feature)
                pruned_tree = pruning(Tree, Tree, test, attr_type, predictor)
                pruned_score = ID3_score(pruned_tree , df[i], attr_type, predictor)
                pruned_score = round((pruned_score*100),2)
                score_list.append(pruned_score)
    return score_list


# ## CART
#CART Node to construct CART Decisio Tree
#    target : predictor value assigned to node
#    parent: parent node for the attribute
#    attribute: attribute node
#    attribute_feature: attribute values
#    children: All the data for the certain child
#    child: Child node of the parents
class CART_Node:
    def __init__(self, target):
        self.target = target
        self.attribute = None
        self.parent = None
        self.children = None
        self.child = {}

# MSE
# CART uses MSE for attribute selection in decision tree. 
def mse(sub_data):
    sub_data = sub_data.values
    return np.mean((sub_data - np.mean(sub_data))**2)

# equal_frequency_split
# It uses equal freqency method to split the points.
# It returns the split points for the dataset.
def equal_frequency_split(df, col, n):
    col_unique =  np.sort(df[col].unique())
    splitting_criteria=[]
    for i in range(n-1):
        splitting_criteria.append(col_unique[int(len(col_unique) / n) * (i+1)])
    return splitting_criteria

# Best_selection
# Using equal_frequency_split function, it finds the splits point for the dataset.
# For each of the split point, it splits the attribute into left node and right node.
# It comptues information gain by fraction of node times mse of the node. 
# Then, it finds the best splits attribute for root node.
def best_selection(df, predictor, freq):
    columns_feature = df.columns[df.columns!=predictor]
    equal_frequency = {}
    for col in columns_feature:
        equal_frequency[col] = equal_frequency_split(df, col, freq)
        
    feature_mse = float('inf')
    feature_root = None
    feature_split = None
    total_gain = 0
    for i in equal_frequency.keys():
        split_freq = equal_frequency[i]
        for j in split_freq:
            left_node = df[df[i]<j]
            left_node_mse = mse(left_node[predictor])

            right_node = df[df[i]>=j]
            right_node_mse = mse(right_node[predictor])

            left_gain = (len(left_node)/len(df))*left_node_mse
            right_gain = (len(right_node)/len(df))*right_node_mse
            total_gain = left_gain + right_gain
            if total_gain < feature_mse:
                feature_mse = total_gain
                feature_root = i
                feature_split = j
    return feature_mse, feature_root, feature_split

#CART
# It select the attribute as root node using best_selection function. 
# The data that is less than the selected binary split is set to the left node, otherwise right node
# Stopping Criteria
#  If every instance has same class label, it returns most unique class
#  If there is no best attribute, then it select the mean of the attribute.
def CART(df, n, predictor):
    if len(df[predictor].unique()) <= 1:
        target_val = df[predictor].unique()[0]
        return CART_Node(target_val)
    else:
        target_val = np.mean(df[predictor])
        TREE = CART_Node(target_val)
        attribute_mse, best_attribute, best_threshold = best_selection(df,predictor, n)
        
        if best_attribute == None:
            target_val = np.mean(df[predictor])
            return CART_Node(target_val)
        TREE.attribute = best_attribute
        df_left = df[df[best_attribute] < best_threshold]
        df_right = df[df[best_attribute] >= best_threshold]
        left_node = df_left.iloc[:, df_left.columns != best_attribute]
        right_node = df_right.iloc[:, df_right.columns != best_attribute]
        
        for i in ['>_' + str(best_threshold), '<_' + str(best_threshold)]:
            sub_data = left_node if i == '>_' + str(best_threshold) else right_node
            subtree = CART(sub_data, n, predictor)
            subtree.parent = best_attribute
            TREE.child[i] = subtree
    return TREE


# ### Early Stopping
#CART_early_stop
# It works very similary compare to normal CART. 
# It select the attribute as root node using best_selection function. 
# The data that is less than the selected binary split is set to the left node, otherwise right node
# Stopping Criteria
#  If every instance has same class label, it returns most unique class
#  If there is no best attribute, then it select the mean of the attribute.
#  If the leaf node is exceed the limit of the leaf node, it stops growing.
#  If there are less than the threshold value, it will not proceed further and returns a mean of the remaining points.  

def CART_early_stop(df, n, min_leaf, threshold, predictor):
    if len(df[predictor].unique()) <= 1:
        target_val = df[predictor].unique()[0]
        return CART_Node(target_val)
    elif len(df) <= min_leaf:
        target_val =np.mean(df[predictor])
        return CART_Node(target_val)
    else:
        target_val = np.mean(df[predictor])
        TREE = CART_Node(target_val)
        attribute_mse, best_attribute, best_threshold = best_selection(df,predictor, n)
        if best_attribute == None:
            target_val = np.mean(df[predictor])
            return CART_Node(target_val)
        elif attribute_mse < threshold:
            return TREE
        else:
            TREE.attribute = best_attribute
            df_left = df[df[best_attribute] < best_threshold]
            df_right = df[df[best_attribute] >= best_threshold]
            left_node = df_left.iloc[:, df_left.columns != best_attribute]
            right_node = df_right.iloc[:, df_right.columns != best_attribute]

            for i in ['>_' + str(best_threshold), '<_' + str(best_threshold)]:
                sub_data = left_node if i == '>_' + str(best_threshold) else right_node
                subtree = CART_early_stop(sub_data, n, min_leaf, threshold, predictor)
                subtree.parent = best_attribute
                subtree.children = sub_data
                TREE.child[i] = subtree
    return TREE

#predict_CART
# if there is no child in the tree, it returns the target value
# if the child node value is smaller than attribute value, then it uses left node, other wise it uses right node.
def predict_CART(df, TREE, row_attr):
    if len(TREE.child) == 0:
        return TREE.target
    else:
        attr_value = row_attr[TREE.attribute]
        tree_child = list(TREE.child.keys())
        if attr_value < float(tree_child [0].split('_')[1]):
            return predict_CART(df, TREE.child[tree_child [0]], row_attr)
        elif attr_value >= float(tree_child[0].split('_')[1]):
            return predict_CART(df, TREE.child[tree_child [1]], row_attr)


# ### CART Score
# CART MSE
# It predicts the predictor value for the dataset, and evaluate the performance using MSE.
# It returns the MSE.
def CART_MSE(TREE, df, predictor):
    pred_list = []
    for i in range(len(df)):
        pred = predict_CART(df, TREE, df.iloc[i, :])
        pred_list.append(pred)     
    mse = evaluation_metrics(df[predictor].values, pred_list, "MSE")
    return mse


# ### CART Evaluation
# CART_eval
#  It computes CART without early-stopping and with early-stopping
#  Unliked it used 80% of dataset for cross-validation for both with early-stopping and without early-stopping.
#  Instead, it used 20% of the dataset for tuning process. 
#  It evaluates the performance using CART_MSE function.
#  It returns the performance of each of the fold. 
def CART_eval(df, predictor, n,  min_leaf, threshold, early_stopping:bool):
    score_list = []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        testing_val_y = df[i][predictor].values
        if early_stopping ==False:
            Tree = CART(k_validate, n, predictor)
            Score = CART_MSE(Tree, df[i], predictor)
            Score = round(Score,2)
            score_list.append(Score)
        else:
            threshold_mse = mse(k_validate[predictor])*threshold
            Tree = CART_early_stop(k_validate, n, min_leaf, threshold_mse, predictor)
            Score = CART_MSE(Tree, df[i], predictor)
            Score = round(Score,2)
            score_list.append(Score)
    return score_list


# ## Print Tree
#Print Tree for ID3
# This is the function to show the structure of decision tree.
# Since it's not tree form, it is hard to picture. 
# Not sure how to visualize this using tree form.
def print_tree(tree):
    for i in tree.child:
        if len(tree.child)!=0:
            if len(tree.child[i].child)==0:
                print(tree.child[i].parent, ' -', i, ' : ', tree.child[i].target)
            else:
                print(tree.child[i].parent, ' -', i)
                print('Subtree')
        print_tree(tree.child[i])

#Print Tree for ID3 pruned (reduced error processing)
# This is the function to show the structure of pruned decision tree.
# Since it's not tree form, it is hard to picture. 
# Not sure how to visualize this using tree form.
def print_tree_prune(tree):
    for i in tree.child:
        if tree.child[i].prune_tag ==False:
            if len(tree.child)!=0:
                if len(tree.child[i].child)==0:
                    print(tree.child[i].parent, ' -', i, ' : ', tree.child[i].target)
                else:
                    print(tree.child[i].parent, ' -', i)
                    print('Subtree')
            print_tree_prune(tree.child[i])


# ## Tuning
# Tuning CART frequency size parameter
# It process tuning to find the frequency size for the CART. 
def tuning_CART(df, frequency_size_candidate, predictor):
    best_candidate = []
    for i in range(len(frequency_size_candidate)):
        best_candidate.append(np.mean(CART_eval(df, predictor, frequency_size_candidate[i], False, False, False)))
    return frequency_size_candidate[best_candidate.index(min(best_candidate))]

# Tuning parameter with frequency size, minimum leaf, and threshold for early-stopping CART algorithm.
def tuning_CART_early(df, frequency_size_candidate,minimum_leaf_candidate, threshold_candidate, predictor):
    best_candidate_early = []
    parameter_list =[]
    for i in range(len(frequency_size_candidate)):
        for j in range(len(minimum_leaf_candidate)):
            for k in range(len(threshold_candidate)):
                parameter_list.append([i,j,k])
                best_candidate_early.append(np.mean(CART_eval(df,
                                                              predictor, 
                                                              frequency_size_candidate[i], 
                                                              minimum_leaf_candidate[j],
                                                              threshold_candidate[k], True)))
    best_parameter =best_candidate_early.index(min(best_candidate_early))
    best_frequency_size = frequency_size_candidate[parameter_list[best_parameter][0]]
    best_minimum_leaf = minimum_leaf_candidate[parameter_list[best_parameter][1]]
    best_threshold = threshold_candidate[parameter_list[best_parameter][2]]
    return best_frequency_size, best_minimum_leaf, best_threshold

# ## Breast Cancer Wisconsin Dataset
print("-------Breast Cancer-------")
breast_cancer_dataset = open_csv_dataset('breast-cancer', True)
breast_cancer_dataset.head(3)
clean_breast_cancer_dataset = handling_missing_values(breast_cancer_dataset)
clean_breast_cancer_dataset.isnull().sum().any()
clean_breast_cancer_dataset.head()
breast_cancer_dataset_v1  = clean_breast_cancer_dataset.copy()
breast_cancer_dataset_v1 = breast_cancer_dataset_v1.drop(['sample_code_number'], axis=1)
breast_cancer_dataset_v1.head(3)
train_breast_cancer_dataset, test_breast_cancer_dataset = split_dataset(breast_cancer_dataset_v1, 0.8)
train_breast_cancer_zscore_dataset, test_breast_cancer_zscore_dataset  = Standardization(train_breast_cancer_dataset,
                                                                                         test_breast_cancer_dataset)
validated_train_breast_cancer = cross_validation(train_breast_cancer_dataset,5)
validated_train_breast_cancer_size = []
for i in range(0, 5):
    validated_train_breast_cancer_size.append(validated_train_breast_cancer[i].shape[0])
#validated_train_breast_cancer_size

#tuning
tuning_breast_cancer = cross_validation(test_breast_cancer_dataset,5)
tuning_breast_cancer_size = []
for i in range(0, 5):
    tuning_breast_cancer_size.append(tuning_breast_cancer [i].shape[0])
#tuning_breast_cancer_size

ID3_score_breast_cancer = ID3(validated_train_breast_cancer, 'class', test_breast_cancer_dataset, 'categorical', False)
ID3_avg_score_breast_cancer = round(np.mean(ID3_score_breast_cancer),4)
print("Breast Cancer Score Unpruned: " + str(ID3_score_breast_cancer) + "/ Avg Score: "+ str(ID3_avg_score_breast_cancer))

#Breast Cancer Score Unpruned: [80.36, 93.69, 77.48, 93.69, 93.69]/ Avg Score: 87.782

ID3_score_breast_cancer_prune = ID3(validated_train_breast_cancer, 'class', test_breast_cancer_dataset, 'categorical', True)
ID3_avg_score_breast_cancer_prune = round(np.mean(ID3_score_breast_cancer_prune),4)
print("Breast Cancer Accuracy Score pruned: " + str(ID3_score_breast_cancer_prune) + "/ Avg Score: "+ str(ID3_avg_score_breast_cancer_prune))

#Breast Cancer Accuracy Score pruned: [73.21, 90.09, 69.37, 87.39, 89.19]/ Avg Score: 81.85


# ## Car Evaluation
print("-------Car-------")
car_dataset = open_csv_dataset('car', True)
car_dataset.head(3)
clean_car_dataset = handling_missing_values(car_dataset)
clean_car_dataset.isnull().sum().any()
clean_car_dataset.head(3)
categorized_car_datast = replacing_string_to_numeric_all_columns(clean_car_dataset)
car_dataset_v1  = categorized_car_datast.copy()
car_dataset_v1.head()
train_car_dataset, test_car_dataset = split_dataset(car_dataset_v1, 0.8)
train_car_zscore_dataset,test_car_zscore_dataset  = Standardization(train_car_dataset,test_car_dataset)
validated_train_car = cross_validation(train_car_dataset,5)

validated_train_car_size = []
for i in range(0, 5):
    validated_train_car_size.append(validated_train_car[i].shape[0])
#validated_train_car_size

tuning_car = cross_validation(test_car_dataset,5)
tuning_car_size = []
for i in range(0, 5):
    tuning_car_size.append(tuning_car[i].shape[0])
#tuning_car_size

ID3_score_car = ID3(validated_train_car, 'class', test_car_dataset, 'categorical', False)
ID3_avg_score_car = round(np.mean(ID3_score_car),4)
print("Car Accuracy Score Unpruned: " + str(ID3_score_car) + "/ Avg Score: "+ str(ID3_avg_score_car))

#Car Accuracy Score Unpruned: [83.03, 71.74, 79.35, 70.65, 69.2]/ Avg Score: 74.794

ID3_score_car_prune = ID3(validated_train_car, 'class', test_car_dataset, 'categorical', True)
ID3_avg_score_car_prune = round(np.mean(ID3_score_car_prune),4)
print("Car Accuracy Score pruned: " + str(ID3_score_car_prune) + "/ Avg Score: "+ str(ID3_avg_score_car_prune))

#Car Accuracy Score pruned: [85.2, 79.35, 66.67, 65.58, 60.14]/ Avg Score: 71.388



# ## Congressional Vote
print("-------Congressional Vote-------")
vote_dataset = open_csv_dataset('vote', True)
vote_dataset.head()
vote_dataset.isnull().sum().any()
categorized_vote_datast = replacing_string_to_numeric_all_columns(vote_dataset)
categorized_vote_datast.head()
vote_dataset_v1  = categorized_vote_datast.copy()
train_vote_dataset, test_vote_dataset = split_dataset(vote_dataset_v1, 0.8)
train_vote_zscore_dataset, test_vote_zscore_dataset  = Standardization(train_vote_dataset,test_vote_dataset)

validated_train_vote = cross_validation(train_vote_dataset,5)
validated_train_vote_size = []
for i in range(0, 5):
    validated_train_vote_size.append(validated_train_vote[i].shape[0])
#validated_train_vote_size

tuning_vote = cross_validation(test_vote_dataset ,5)
tuning_vote_size = []
for i in range(0, 5):
    tuning_vote_size.append(tuning_vote[i].shape[0])
#tuning_vote_size

ID3_score_vote = ID3(validated_train_vote, 'class', test_vote_dataset, 'categorical', False)
ID3_avg_score_vote = round(np.mean(ID3_score_vote),4)
print("vote Accuracy Score Unpruned: " + str(ID3_score_vote) + "/ Avg Score: "+ str(ID3_avg_score_vote))

#vote Accuracy Score Unpruned: [97.14, 92.86, 94.29, 92.86, 94.12]/ Avg Score: 94.254

ID3_score_vote_prune = ID3(validated_train_vote, 'class', test_vote_dataset, 'categorical', True)
ID3_avg_score_vote_prune = round(np.mean(ID3_score_vote_prune),4)
print("vote Accuracy Score pruned: " + str(ID3_score_vote_prune) + "/ Avg Score: "+ str(ID3_avg_score_vote_prune))

#vote Accuracy Score pruned: [97.14, 92.86, 94.29, 95.71, 92.65]/ Avg Score: 94.53


# ## Abalone

print("-------Abalone-------")
abalone_dataset = open_csv_dataset('abalone', True)
abalone_dataset.head(3)
abalone_dataset.isnull().sum().any()
abalone_dataset = abalone_dataset.drop(['sex'], axis = 1)
abalone_dataset_v1  = abalone_dataset.copy()
abalone_discretize_dataset = abalone_dataset_v1.copy()
abalone_discretize_dataset = discretization(abalone_discretize_dataset, 5, 'equal_frequency',
                                           ['length', 'diameter', 'height', 'whole_height',
                                            'shucked_height', 'viscera_weight', 'shell_weight', 'rings'])
abalone_dataset_v2 =  abalone_dataset_v1.copy()

train_abalone_dataset, test_abalone_dataset = split_dataset(abalone_dataset_v2, 0.8)
train_abalone_zscore_dataset,test_abalone_zscore_dataset  = Standardization(train_abalone_dataset,
                                                                             test_abalone_dataset)

validated_train_abalone = cross_validation_regression(train_abalone_dataset, 5, 'rings')
validated_train_abalone_size= []
for i in range(0, 5):
    validated_train_abalone_size.append(validated_train_abalone[i].shape[0])
#validated_train_abalone_size
tuning_abalone = cross_validation_regression(test_abalone_dataset, 5, 'rings')
tuning_abalone_size = []
for i in range(0, 5):
    tuning_abalone_size.append(tuning_abalone[i].shape[0])
#tuning_abalone_size
frequency_size_candidate = [i for i in range(2, 10)]
minimum_leaf_candidate = [i for i in range(2, 10)]
threshold_candidate = [0.2, 0.4, 0.6, 0.8]
frequency_cart = tuning_CART(tuning_abalone, frequency_size_candidate, 'rings')
# frequency_cart= 2
# frequency_cart_early, minimum_leaf_early, threshold_early = 2,6,0.2
print("Frequency Size for Cart: " +  str(frequency_cart))

frequency_cart_early, minimum_leaf_early, threshold_early = tuning_CART_early(tuning_abalone, 
                                                                              frequency_size_candidate, 
                                                                              minimum_leaf_candidate, 
                                                                              threshold_candidate, 
                                                                              'rings')

print("Frequency Size for Cart Early Stopping: " +  str(frequency_cart_early))
print("Minimum Leaf Size for Cart Early Stopping: " +  str(minimum_leaf_early))
print("Threshold for Cart Early Stopping: " +  str(threshold_early))

cart_score_abalone = CART_eval(validated_train_abalone, 'rings', frequency_cart, False, False, False)
cart_avg_score_abalone  = round(np.mean(cart_score_abalone),2)
print("Abalone MSE Unpruned: " + str(cart_score_abalone) + "/ Avg Score: "+ str(cart_avg_score_abalone))
#Abalone MSE Unpruned: [10.86, 6.04, 4.84, 4.45, 35.08]/ Avg Score: 12.25

cart_score_abalone_early  = CART_eval(validated_train_abalone,'rings', frequency_cart_early, minimum_leaf_early, threshold_early, True)
cart_avg_score_abalone_early = round(np.mean(cart_score_abalone_early),2)
print("Abalone MSE Pruned: " + str(cart_score_abalone_early) + "/ Avg Score: "+ str(cart_avg_score_abalone_early))
#Abalone MSE Pruned: [11.78, 6.15, 5.09, 4.54, 35.09]/ Avg Score: 12.53


# ## Computer Hardware

print("-------Computer Hardware-------")
computer_dataset = open_csv_dataset('machine', True)
computer_dataset.head()
computer_dataset.isnull().sum().any()
computer_ERP  = computer_dataset['ERP']
computer_dataset = computer_dataset.drop(['vendor', 'model', 'ERP'], axis = 1)
computer_dataset_v1  = computer_dataset.copy()
computer_dataset_v1.head()
computer_dataset_v1['PRP'] =computer_dataset_v1['PRP'].apply(computer_categorical_distribution)
computer_discretize_dataset = computer_dataset_v1.copy()
computer_discretize_dataset = discretization(computer_discretize_dataset, 
                                             5, 
                                             'equal_frequency', 
                                             ['myct','mmin', 'mmax', 'cach', 'chmin', 'chmax'])
computer_dataset_v2 = computer_dataset_v1.copy()

train_computer_dataset, test_computer_dataset = split_dataset(computer_dataset_v2, 0.8)
train_computer_zscore_dataset, test_computer_zscore_dataset  = Standardization(train_computer_dataset,
                                                                               test_computer_dataset)
validated_train_computer = cross_validation_regression(train_computer_dataset, 5, 'PRP')
validated_train_computer_size = []
for i in range(0, 5):
    validated_train_computer_size.append(validated_train_computer[i].shape[0])
#validated_train_computer_size

tuning_computer = cross_validation_regression(test_computer_dataset , 5, 'PRP')
tuning_computer_size = []
for i in range(0, 5):
    tuning_computer_size.append(tuning_computer[i].shape[0])
#tuning_computer_size

frequency_size_candidate = [i for i in range(2, 10)]
minimum_leaf_candidate = [i for i in range(2, 10)]
threshold_candidate = [0.2, 0.4, 0.6, 0.8]

frequency_cart = tuning_CART(tuning_computer , frequency_size_candidate, 'PRP')
# frequency_cart= 3
# frequency_cart_early, minimum_leaf_early, threshold_early = 4,2,0.2
print("Frequency Size for Cart: " +  str(frequency_cart))

frequency_cart_early, minimum_leaf_early, threshold_early   = tuning_CART_early(tuning_computer , 
                                                                                 frequency_size_candidate, 
                                                                                 minimum_leaf_candidate, 
                                                                                 threshold_candidate, 
                                                                                 'PRP')
print("Frequency Size for Cart Early Stopping: " +  str(frequency_cart_early))
print("Minimum Leaf Size for Cart Early Stopping: " +  str(minimum_leaf_early))
print("Threshold for Cart Early Stopping: " +  str(threshold_early))

cart_score_computer = CART_eval(validated_train_computer ,'PRP', frequency_cart, False, False, False)
cart_avg_score_computer  = round(np.mean(cart_score_computer),2)
print("Computer MSE Unpruned: " + str(cart_score_computer) + "/ Avg Score: "+ str(cart_avg_score_computer ))
#Computer MSE Unpruned: [0.8, 0.16, 0.23, 0.48, 5.35]/ Avg Score: 1.4

cart_score_computer_early = CART_eval(validated_train_computer , 'PRP', frequency_cart_early, minimum_leaf_early, threshold_early, True)
cart_avg_score_computer_early  = round(np.mean(cart_score_computer_early),2)
print("Computer MSE Pruned: " + str(cart_score_computer_early) + "/ Avg Score: "+ str(cart_avg_score_computer_early))
#Computer MSE Pruned: [1.16, 0.21, 0.27, 0.75, 5.09]/ Avg Score: 1.5


# ## Forest Fires
print("-------Forest Fires-------")
forest_dataset = open_csv_dataset('forest', False)
forest_dataset.isnull().sum().any()
categorized_forest_dataset = replacing_string_to_numeric_multiple_columns(forest_dataset,['month', 'day'])
categorized_forest_dataset.head()
forest_dataset_v1  = categorized_forest_dataset.copy()
forest_dataset_v1['area'] = log_transform(forest_dataset_v1['area'])
forest_dataset_v1.head()
forest_discretize_dataset = forest_dataset_v1.copy()
forest_discretize_dataset = discretization(forest_discretize_dataset, 
                                           5, 
                                           'equal_frequency', 
                                           ['FFMC', 'DMC', 'DC', 'ISI','temp','RH', 'wind', 'rain'])

train_forest_dataset, test_forest_dataset = split_dataset(forest_dataset_v1, 0.8)
train_forest_zscore_dataset, test_forest_zscore_dataset  = Standardization(train_forest_dataset,test_forest_dataset)

validated_train_forest = cross_validation_regression(train_forest_dataset, 5, 'area')
validated_train_forest_size  = []
for i in range(0, 5):
    validated_train_forest_size.append(validated_train_forest[i].shape[0])
#validated_train_forest_size 

tuning_forest = cross_validation_regression(test_forest_dataset, 5, 'area')
tuning_forest_size  = []
for i in range(0, 5):
    tuning_forest_size .append(tuning_forest[i].shape[0])
#tuning_forest_size 

frequency_size_candidate = [i for i in range(2, 10)]
minimum_leaf_candidate = [i for i in range(2, 10)]
threshold_candidate = [0.2, 0.4, 0.6, 0.8]

frequency_cart = tuning_CART(tuning_forest ,frequency_size_candidate, 'area')
# frequency_cart= 9
# frequency_cart_early, minimum_leaf_early, threshold_early = 2,7,0.8
print("Frequency Size for Cart: " +  str(frequency_cart))

frequency_cart_early, minimum_leaf_early, threshold_early   = tuning_CART_early(tuning_forest, 
                                                                                 frequency_size_candidate, 
                                                                                 minimum_leaf_candidate, 
                                                                                 threshold_candidate, 
                                                                                 'area')
print("Frequency Size for Cart Early Stopping: " +  str(frequency_cart_early))
print("Minimum Leaf Size for Cart Early Stopping: " +  str(minimum_leaf_early))
print("Threshold for Cart Early Stopping: " +  str(threshold_early))

cart_score_forest = CART_eval(validated_train_forest, 'area', frequency_cart, False, False, False)
cart_avg_score_forest = round(np.mean(cart_score_forest),2)
print("Forest MSE Unpruned: " + str(cart_score_forest) + "/ Avg Score: "+ str(cart_avg_score_forest))
#Forest MSE Unpruned: [3.68, 4.08, 2.83, 1.32, 8.0]/ Avg Score: 3.98


cart_score_forest_early = CART_eval(validated_train_forest , 'area', frequency_cart_early, minimum_leaf_early, threshold_early, True)
cart_avg_score_forest_early = round(np.mean(cart_score_forest_early),2)
print("Forest MSE Pruned: " + str(cart_score_forest_early) + "/ Avg Score: "+ str(cart_avg_score_forest_early ))
#Forest MSE Pruned: [2.26, 2.72, 1.15, 0.95, 8.37]/ Avg Score: 3.09



