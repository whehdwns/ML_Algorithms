# -*- coding: utf-8 -*-
"""Programing_Project_2_DCho.ipynb
"""

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import os
import glob
from collections import Counter # Mode
import warnings
warnings.filterwarnings("ignore")
# from google.colab import drive
# drive.mount('/content/drive')

path = os.getcwd()
all_files = sorted(glob.glob(path + "/dataset/*.data"))
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

"""## Loading Data"""

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

"""## Handling Missing Values"""

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

"""## Handling Categorical Data"""

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

"""## Categorical Distribution for Computer Hardware Dataset"""

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

"""## Log Transform"""

# Log Transform
# 1. This function is used to apply for Forest Fires data. 
# 2. Based on note, it shows the output area is very skewed toward 0.0. The authors recommend a log transform.
# 3. It log transform certain columns in the dataset. 

def log_transform(x):
    return np.log(x + 1)

"""## Discretization"""

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

"""## Standardization"""

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

"""## Cross-validation"""

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

"""## Evaluation Metrics"""

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

"""## Euclidean Distance"""

# It computes the distance using Euclidean Distance formula.
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

"""## Gaussian Kernel"""

# It computes the smooth function using Gaussian Kernel Function
def gaussian_kernel(distances, sigma):
    return np.exp(-(1 / 2 * sigma) * distances)

"""## K Nearest Neighbor Classifier (Standard KNN, Edited KNN, Condensed KNN)"""

# K-Nearest Neighbor Algorithm Classifier
# 1. It computes the distance between the instance label and the training instance using eucliden distance.
# 2. It sorts the distance by ascending order
# 3. It checks for k closest neighbor. 
# 4. From the k closest neighbor, it takes most common label using plurality vote to determine the class.
# 5. It returns the list of the prediction.

#For demonstration purpose
# It returns with k-nearest neighbor and assigned class.

def knn_classifer(train, test, predictor, k):
    train_class = train[predictor].values
    train = train.drop(columns=[predictor]).values
    test_class = test[predictor].values
    test = test.drop(columns=[predictor]).values
    prediction = []
    Nearest_neighbor =[]
    assigned_class=[]
    for n in test:
        distances = []
        for j in train:
            distances.append(euclidean_distance(n,j))
        indices = np.argsort(distances, axis=0)
        Nearest_neighbor.append(indices[:k])
        labels = train_class[indices[:k]]
        assigned_class.append(labels)
        most_labels = Counter(labels).most_common(1)
        prediction.append(most_labels[0][0])
    return prediction, Nearest_neighbor, assigned_class

# Edited K Nearest Neighbor Classifier
# 1. Each of the training points is classified by iterating all of the other data points in the data.
# 2. It comptues the prediction using standard KNN.
# 3. It compares the classification returned to the label associated with the point
# 4. If it disagree, it drop/removes from dataframe.
# 5. It computes the performance by evaluating accurasy score. 
# 6. It repeats the process until the performance doesn't improve. 
# 7. It returns the remaining dataframe. 

def edited_knn_classifer(df, test, predictor):
    edited_df = df.copy()
    val_labels = test[predictor].values
    new_score = 0
    pre_score = -1
    while new_score > pre_score:
        pre_knn = knn_classifer(edited_df, test, predictor, 1)[0]
        pre_score = evaluation_metrics(val_labels, pre_knn, "classification score")
        for ind, row in edited_df.iterrows():
            changed_df = edited_df.loc[lambda x: x.index != ind]
            pred = knn_classifer(changed_df, row.to_frame().T, predictor, 1)[0]
            true_label = row[predictor]
            if pred !=true_label:
                edited_df.drop(ind, inplace=True)
        new_knn = knn_classifer(edited_df, test, predictor, 1)[0]
        new_score = evaluation_metrics(val_labels, new_knn, "classification score")
    return edited_df

# Condensed K Nearest Neighbor Classifier
# 1. Condensed KNN uses similar method as Edited KNN.
# 2. It starts from empty set Z and passes over the instance in random order and computes the prediction using standard KNN.
# 3. It finds the points within set of Z that is minimum distance to each points.
# 4. Since it finds the minimum distance, it uses 1 for number of neighbor.
# 5. If an instance is misclassified, the points added into Z.
# 6. It repeats the process until Z doesn't change. 
# 7. It returns the set of Z.

def condensed_knn_classifer(df, predictor):
    condensed_df = df.copy()
    Z = []
    condensed_df = condensed_df.sample(frac=1)
    changed = True
    while changed:
        changed =False
        for indexed in range(condensed_df.shape[0]):
            if len(Z)==0:
                Z.append(indexed)
                changed = True
            if indexed in Z:
                continue
            new_df = condensed_df.iloc[Z]
            pred = knn_classifer(new_df, condensed_df.iloc[indexed].to_frame().T, predictor, 1)[0]
            true_label = condensed_df.iloc[indexed][predictor]
            if pred != true_label:
                Z.append(indexed)
                changed =True  
    new_df =condensed_df.iloc[Z]
    return new_df

# Knn_classification
# It computes various KNN algorithm for classifier.
# It evalutes the performance using accuracy score.
# It returns the performance and remaining data after Edited and Condensed KNN. 

def knn_classification(df, predictor, k, eval_method, algorithm):
    k_fold_score = []
    remaining_data = []
    knn_demonstration= []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        testing_val_y = df[i][predictor].values
        if algorithm =="knn":
            y_pred = knn_classifer(k_validate, df[i], predictor, k)[0]
            pred, closest_neighbor, assigned_class = knn_classifer(k_validate, df[i], predictor, k)
            knn_demonstration = [pred, closest_neighbor, assigned_class]
        if algorithm =="edited knn":
            edited_knn_df = edited_knn_classifer(k_validate,df[i], predictor)
            y_pred = knn_classifer(edited_knn_df, df[i], predictor, k)[0]
            remaining_data.append(round((len(edited_knn_df)/len(k_validate))*100,2))
        if algorithm == "condensed knn":
            condensed_knn_df = condensed_knn_classifer(k_validate, predictor)
            y_pred = knn_classifer(condensed_knn_df, df[i], predictor, k)[0]
            remaining_data.append(round((len(condensed_knn_df)/len(k_validate))*100,2))

        score = evaluation_metrics(testing_val_y, y_pred, eval_method)
        k_fold_score.append(round((score*100),2))
    return k_fold_score, remaining_data, knn_demonstration

"""## K Nearest Neighbor Regressor (Standard KNN, Edited KNN, Condensed KNN)"""

# K Nearest Neighbor Regressor
# 1. It computes the distance between the instance label and the training instance using eucliden distance.
# 2. It sorts the distance by ascending order
# 3. It checks for k closest neighbor. 
# 4. It applies a Gaussian (radial basis function) kernel to computes the weights.
# 5. It makes prediction by calculating regression values. {Sum(weight * Nearest Neighbor) / Sum(weight)}

def knn_regressor(train, test, predictor, k, sigma):
    train_class = train[predictor].values
    train = train.drop(columns=[predictor]).values
    test_class = test[predictor].values
    test = test.drop(columns=[predictor]).values
    prediction = []
    Nearest_neighbor =[]
    assigned_class=[]
    for n in test:
        distances = []
        for x_train in train:
            distances.append(euclidean_distance(n,x_train))
        indices = np.argsort(distances, axis=0)
        Nearest_neighbor.append(indices[:k])
        labels = train_class[indices[:k]]
        assigned_class.append(labels)
        nearest_distance = np.array(distances)[indices[:k]]
        weight = gaussian_kernel(nearest_distance, sigma)
        nearest_neighbor = train_class[indices[:k]]
        prediction.append(sum(weight*nearest_neighbor)/ sum(weight))
    return prediction, Nearest_neighbor, assigned_class

# Edited K Nearest Neighbor Regressor
# 1. It comptues the prediction using standard KNN.
# 2. It compares the classification returned to the label associated with the point
# 3. If the prediction doesn't match the rest of the data, it drops the points from the training set
# 4. It checks that it doesn't matches when the prediction is not within error threshold.
# 5. Since it compares previous score and current score for MSE, the score should be minimum.
# 6. It repeats the process until it doesn't reduce the MSE score.

def edited_knn_regressor(df, test, predictor, sigma, epsilon):
    edited_df = df.copy()
    val_labels = test[predictor].values
    new_score = -1
    pre_score = 0
    while new_score < pre_score:
        pre_knn = knn_regressor(edited_df, test, predictor, 1, sigma)[0]
        pre_score = evaluation_metrics(val_labels, pre_knn, "MSE")
        for ind, row in edited_df.iterrows():
            remaining_data = edited_df.loc[lambda x: x.index != ind]
            pred = knn_regressor(remaining_data, row.to_frame().T, predictor, 1, sigma)[0]
            true_values = row[predictor]
            if not((true_values - epsilon) < pred < (true_values + epsilon)):
                edited_df.drop(ind, inplace=True)
        new_knn = knn_regressor(edited_df, test, predictor, 1, sigma)[0]
        new_score = evaluation_metrics(val_labels, new_knn, "MSE")
    return edited_df

# Condensed K Nearest Neighbor for Regressor
# 1. It starts from empty set Z.
# 2. It adds inital points when Z is empty
# 3. It makes prediction by using standard KNN regressor.
# 4. If the prediction is not within error threshold, it adds to Z
# 5. It returns the set of Z

def condensed_knn_regressor(df, predictor, sigma, epsilon):
    df2 = df.copy()
    Z = []
    df2 = df2.sample(frac=1)
    changed = True
    while changed:
        changed =False
        for indexed in range(df2.shape[0]):
            if len(Z)==0:
                Z.append(indexed)
                changed = True
            if indexed in Z:
                continue
            new_df = df2.iloc[Z]
            pred = knn_regressor(new_df, df2.iloc[indexed].to_frame().T, predictor, 1, sigma)[0]
            true_values = df2.iloc[indexed][predictor]
            if not((true_values - epsilon) < pred < (true_values + epsilon)):
                Z.append(indexed)
                changed =True  
    new_df =df.iloc[Z]
    return new_df

# KNN Regression
# It computes various KNN algorithm for regressor
# It evalutes the performance using MSE.
# It returns the performance and remaining data after Edited and Condensed KNN.

def knn_regression(df, predictor, k, sigma, epsilon, eval_method, algorithm):
    k_fold_score = []
    remaining_data = []
    knn_demonstration = []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        testing_val_y = df[i][predictor].values
        if algorithm =="knn":
            predicted = knn_regressor(k_validate, df[i], predictor, k, sigma)[0]
            pred, closest_neighbor, asigned_class = knn_regressor(k_validate, df[i], predictor, k, sigma)
            knn_demonstration = [pred, closest_neighbor, asigned_class]
        if algorithm =="edited knn":
            edited_knn_df = edited_knn_regressor(k_validate,df[i], predictor, sigma, epsilon)
            predicted = knn_regressor(edited_knn_df, df[i], predictor, k, sigma)[0]
            remaining_data.append(round((len(edited_knn_df)/len(k_validate))*100,2))
        if algorithm =="condensed knn":
            condensed_knn_df = condensed_knn_regressor(k_validate, predictor, sigma, epsilon)
            predicted = knn_regressor(condensed_knn_df, df[i], predictor, k, sigma)[0]
            remaining_data.append(round((len(condensed_knn_df)/len(k_validate))*100,2))
        score = evaluation_metrics(testing_val_y, predicted, eval_method)
        k_fold_score.append(round(score,4))
    return k_fold_score, remaining_data, knn_demonstration

"""## KNN demonstration classification/Regression"""

# This is the demonstration of neighbors returns as well as the point being classified.
# It showed closest Neighbor and k assigned class and predicted class.

def knn_demonstration_classification(df, predictor, k):
    knn_demo = knn_classification(df, predictor, k, "classification score", "knn")[2]
    df2 = pd.DataFrame()
    df2['closest Neighbor '] = knn_demo[1]
    df2['Assigned Class'] = knn_demo[2]
    df2['Predicted Class'] = knn_demo[0]
    return df2

def knn_demonstration_regression(df, predictor, k, sigma):
    knn_demo = knn_regression(df, predictor, k , sigma, False, "MSE", 'knn')[2]
    df2 = pd.DataFrame()
    df2['closest Neighbor '] = knn_demo[1]
    df2['Assigned Class'] = knn_demo[2]
    df2['Predicted Class'] = knn_demo[0]
    return df2

"""## Tuning"""

# Tuning k parameter
# It process tuning to find the best k parameter for the algorithm.

def tuning_k_parameter(df, parameter, predictor, eval_method, algorithm):
    best_k_candidate = []
    for i in range(len(parameter)):
        best_k_candidate.append(np.mean(knn_classification(df, predictor, parameter[i], eval_method, algorithm)[0]))
    return parameter[best_k_candidate.index(max(best_k_candidate))]

# Tuning parameter with sigma
# It process tuning to find the best k and sigma for the algorithm.

def tuning_parameter_with_sigma(df, kparameter, sigmaparameter, epsilon, predictor, eval_method, algorithm):
    parameter_list =[]
    best_candidate = []
    for i in range(len(kparameter)):
        for j in range(len(sigmaparameter)):
            parameter_list.append([i,j])
            best_candidate.append(np.mean(knn_regression(df, predictor,kparameter[i], sigmaparameter[j],False, eval_method, algorithm)[0]))
    best_parameter =best_candidate.index(min(best_candidate))
    best_k = kparameter[parameter_list[best_parameter][0]]
    best_sigma = sigmaparameter[parameter_list[best_parameter][1]]
    return best_k, best_sigma

# Tuning parameter with epsilon
# It process tuning to find the best k, sigma, and epsilon for the algorithm.

def tuning_parameter_with_epsilon(df, kparameter, sigmaparameter, epsilonparameter, predictor, eval_method, algorithm):
    parameter_list =[]
    best_candidate = []
    for i in range(len(kparameter)):
        for j in range(len(sigmaparameter)):
            for k in range(len(epsilonparameter)):
                parameter_list.append([i,j,k])
                best_candidate.append(np.mean(knn_regression(df, predictor,kparameter[i], sigmaparameter[j],epsilonparameter[k], eval_method, algorithm)[0]))
    best_parameter =best_candidate.index(min(best_candidate))
    best_k = kparameter[parameter_list[best_parameter][0]]
    best_sigma = sigmaparameter[parameter_list[best_parameter][1]]
    best_epsilon = epsilonparameter[parameter_list[best_parameter][2]]
    return best_k, best_sigma,  best_epsilon

"""## Breast Cancer Wisconsin Dataset"""
print("-------Breast Cancer-------")
breast_cancer_dataset = open_csv_dataset('breast-cancer', True)
clean_breast_cancer_dataset = handling_missing_values(breast_cancer_dataset)
clean_breast_cancer_dataset.isnull().sum().any()
breast_cancer_dataset_v1  = clean_breast_cancer_dataset.copy()
breast_cancer_dataset_v1 = breast_cancer_dataset_v1.drop(['sample_code_number'], axis=1)
train_breast_cancer_dataset, test_breast_cancer_dataset = split_dataset(breast_cancer_dataset_v1, 0.8)

train_breast_cancer_zscore_dataset, test_breast_cancer_zscore_dataset  = Standardization(train_breast_cancer_dataset,
                                                                                         test_breast_cancer_dataset)
validated_train_breast_cancer = cross_validation(train_breast_cancer_dataset,5)

validated_train_breast_cancer_size = []
for i in range(0, 5):
    validated_train_breast_cancer_size.append(validated_train_breast_cancer[i].shape[0])
print("Size of Cross Validation: " + str(validated_train_breast_cancer_size))

#tuning
tuning_breast_cancer = cross_validation(test_breast_cancer_dataset,5)
tuning_breast_cancer_size = []
for i in range(0, 5):
    tuning_breast_cancer_size.append(tuning_breast_cancer [i].shape[0])
print("Size of tuning: " + str(tuning_breast_cancer_size))

k_values = [i for i in range(1, 11)]
best_knn_k_breast_cancer = tuning_k_parameter(tuning_breast_cancer, k_values, 'class', "classification score", "knn")
best_eknn_k_breast_cancer = tuning_k_parameter(tuning_breast_cancer, k_values, 'class', "classification score", "edited knn")
best_cknn_k_breast_cancer =  tuning_k_parameter(tuning_breast_cancer, k_values, 'class', "classification score", "condensed knn")

print("Best KNN parameter: "+str(best_knn_k_breast_cancer))
print("Best Edited KNN parameter: " + str(best_eknn_k_breast_cancer))
print("Best Condensed KNN parameter: " + str(best_cknn_k_breast_cancer))

knn_demonstration_classification(validated_train_breast_cancer, 'class', best_knn_k_breast_cancer).head()

knn_score_breast_cancer = knn_classification(validated_train_breast_cancer, 'class', best_knn_k_breast_cancer, "classification score", "knn")[0]
knn_avg_score_breast_cancer = round(np.mean(knn_score_breast_cancer),2)
print("Accuracy Scores for KNN: " + str(knn_score_breast_cancer) + "/ Avg Score: "+ str(knn_avg_score_breast_cancer))

eknn_score_breast_cancer, eknn_remaining_breast_cancer = knn_classification(validated_train_breast_cancer, 'class', best_eknn_k_breast_cancer, "classification score", "edited knn")[:2]
eknn_avg_score_breast_cancer = round(np.mean(eknn_score_breast_cancer),2)
print("Remaining Data % after Edited KNN: "+ str(eknn_remaining_breast_cancer) + " / Avg Remaining Data %: "+ str(round(np.mean(eknn_remaining_breast_cancer),2)))
print("Accuracy Scores for Edited KNN: " + str(eknn_score_breast_cancer) + " / Avg Score: "+ str(eknn_avg_score_breast_cancer))

cknn_score_breast_cancer, cknn_remaining_breast_cancer = knn_classification(validated_train_breast_cancer, 'class', best_cknn_k_breast_cancer, "classification score", "condensed knn")[:2]
cknn_avg_score_breast_cancer = round(np.mean(cknn_score_breast_cancer),2)
print("Remaining Data % after Condensed KNN: "+ str(cknn_remaining_breast_cancer)+ " / Avg Remaining Data %: "+ str(round(np.mean(cknn_remaining_breast_cancer),2)))
print("Accuracy Scores for Condensed KNN: " + str(cknn_score_breast_cancer) + " / Avg Score: "+ str(cknn_avg_score_breast_cancer))

"""## Car Evaluation"""
print("-------Car Evaluation-------")
car_dataset = open_csv_dataset('car', True)
clean_car_dataset = handling_missing_values(car_dataset)
clean_car_dataset.isnull().sum().any()
categorized_car_datast = replacing_string_to_numeric_all_columns(clean_car_dataset)
car_dataset_v1  = categorized_car_datast.copy()
train_car_dataset, test_car_dataset = split_dataset(car_dataset_v1, 0.8)
train_car_zscore_dataset,test_car_zscore_dataset  = Standardization(train_car_dataset,test_car_dataset)

validated_train_car = cross_validation(train_car_dataset,5)

validated_train_car_size = []
for i in range(0, 5):
    validated_train_car_size.append(validated_train_car[i].shape[0])
print("Size of Cross Validation: " + str(validated_train_car_size))

tuning_car = cross_validation(test_car_dataset,5)

tuning_car_size = []
for i in range(0, 5):
    tuning_car_size.append(tuning_car[i].shape[0])
print("Size of tuning: " + str(tuning_car_size))

k_values = [i for i in range(1, 11)]
best_knn_k_car = tuning_k_parameter(tuning_car, k_values, 'class', "classification score", "knn")
best_eknn_k_car = tuning_k_parameter(tuning_car, k_values, 'class', "classification score", "edited knn")
best_cknn_k_car =  tuning_k_parameter(tuning_car, k_values, 'class', "classification score", "condensed knn")

print("Best KNN parameter: "+str(best_knn_k_car))
print("Best Edited KNN parameter: " + str(best_eknn_k_car))
print("Best Condensed KNN parameter: " + str(best_cknn_k_car))

knn_score_car = knn_classification(validated_train_car, 'class', best_knn_k_car, "classification score", "knn")[0]
knn_avg_score_car = round(np.mean(knn_score_car),2)
print("Accuracy Scores for KNN: " + str(knn_score_car) + "/ Avg Score: "+ str(knn_avg_score_car))

eknn_score_car, eknn_remaining_car = knn_classification(validated_train_car, 'class', best_eknn_k_car, "classification score", "edited knn")[:2]
eknn_avg_score_car = round(np.mean(eknn_score_car),2)
print("Remaining Data % after Edited KNN: "+ str(eknn_remaining_car)+ " / Avg Remaining Data %: "+ str(round(np.mean(eknn_remaining_car),2)))
print("Accuracy Scores for Edited KNN: " + str(eknn_score_car) + " / Avg Score: "+ str(eknn_avg_score_car))

cknn_score_car, cknn_remaining_car = knn_classification(validated_train_car, 'class', best_cknn_k_car, "classification score", "condensed knn")[:2]
cknn_avg_score_car = round(np.mean(cknn_score_car),2)
print("Remaining Data % after Condensed KNN: "+ str(cknn_remaining_car)+ " / Avg Remaining Data %: "+ str(round(np.mean(cknn_remaining_car),2)))
print("Accuracy Scores for Condensed KNN: " + str(cknn_score_car) + " / Avg Score: "+ str(cknn_avg_score_car))

"""## Congressional Vote"""
print("-------Congressional Vote-------")
vote_dataset = open_csv_dataset('vote', True)
vote_dataset.isnull().sum().any()
categorized_vote_datast = replacing_string_to_numeric_all_columns(vote_dataset)
vote_dataset_v1  = categorized_vote_datast.copy()
train_vote_dataset, test_vote_dataset = split_dataset(vote_dataset_v1, 0.8)

train_vote_zscore_dataset, test_vote_zscore_dataset  = Standardization(train_vote_dataset,test_vote_dataset)

validated_train_vote = cross_validation(train_vote_dataset,5)

validated_train_vote_size = []
for i in range(0, 5):
    validated_train_vote_size.append(validated_train_vote[i].shape[0])
print("Size of Cross Validation: " + str(validated_train_vote_size))

tuning_vote = cross_validation(test_vote_dataset ,5)

tuning_vote_size = []
for i in range(0, 5):
    tuning_vote_size.append(tuning_vote[i].shape[0])
print("Size of tuning: " + str(tuning_vote_size))

k_values = [i for i in range(1, 11)]
best_knn_k_vote = tuning_k_parameter(tuning_vote, k_values, 'class', "classification score", "knn")
best_eknn_k_vote = tuning_k_parameter(tuning_vote, k_values, 'class', "classification score", "edited knn")
best_cknn_k_vote =  tuning_k_parameter(tuning_vote, k_values, 'class', "classification score", "condensed knn")

print("Best KNN parameter: "+str(best_knn_k_vote))
print("Best Edited KNN parameter: " + str(best_eknn_k_vote))
print("Best Condensed KNN parameter: " + str(best_cknn_k_vote))

knn_score_vote = knn_classification(validated_train_vote, 'class', best_knn_k_vote, "classification score", "knn")[0]
knn_avg_score_vote = round(np.mean(knn_score_vote),2)
print("Accuracy Scores for KNN: " + str(knn_score_vote) + "/ Avg Score: "+ str(knn_avg_score_vote))

eknn_score_vote, eknn_remaining_vote = knn_classification(validated_train_vote, 'class', best_eknn_k_vote, "classification score", "edited knn")[:2]
eknn_avg_score_vote = round(np.mean(eknn_score_vote),2)
print("Remaining Data % after Edited KNN: "+ str(eknn_remaining_vote)+ " / Avg Remaining Data %: "+ str(round(np.mean(eknn_remaining_vote),2)))
print("Accuracy Scores for Edited KNN: " + str(eknn_score_vote) + " / Avg Score: "+ str(eknn_avg_score_vote))

cknn_score_vote, cknn_remaining_vote = knn_classification(validated_train_vote, 'class', best_cknn_k_vote, "classification score", "condensed knn")[:2]
cknn_avg_score_vote = round(np.mean(cknn_score_vote),2)
print("Remaining Data % after Condensed KNN: "+ str(cknn_remaining_vote)+ " / Avg Remaining Data %: "+ str(round(np.mean(cknn_remaining_vote),2)))
print("Accuracy Scores for Condensed KNN: " + str(cknn_score_vote) + " / Avg Score: "+ str(cknn_avg_score_vote))

"""## Abalone"""
print("-------Abalone-------")
abalone_dataset = open_csv_dataset('abalone', True)
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
print("Size of Cross Validation: " + str(validated_train_abalone_size))
tuning_abalone = cross_validation_regression(test_abalone_dataset, 5, 'rings')
tuning_abalone_size = []
for i in range(0, 5):
    tuning_abalone_size.append(tuning_abalone[i].shape[0])
print("Size of tuning: " + str(tuning_abalone_size))

# k_values = [i for i in range(1, 11)]
# sigma_values =[0.01, 0.05, 0.1, 0.5, 1, 2]
# epsilon_values = [0.01, 0.05, 0.1, 0.5, 1]

# best_knn_k_abalone, best_knn_sigma_abalone = tuning_parameter_with_sigma(tuning_abalone, k_values, 
#                                                                            sigma_values, False, 
#                                                                            'rings', 'MSE', 'knn')
# best_eknn_k_abalone, best_eknn_sigma_abalone, best_eknn_epsilon_abalone= tuning_parameter_with_epsilon(tuning_abalone, k_values, sigma_values,epsilon_values, 'rings', 'MSE', 'edited knn')
# best_cknn_k_abalone, best_cknn_sigma_abalone, best_cknn_epsilon_abalone = tuning_parameter_with_epsilon(tuning_abalone, k_values, sigma_values,epsilon_values, 'rings', 'MSE', 'condensed knn')

# print("Best KNN parameter:  Best K: "+str(best_knn_k_abalone) + " / Best Sigma: "+str(best_knn_sigma_abalone))
# print("Best Edited KNN parameter: Best K: "+str(best_eknn_k_abalone) + " / Best Sigma: "+str(best_eknn_sigma_abalone)+" / Best Epsilon: "+str(best_eknn_epsilon_abalone))
# print("Best Condensed KNN parameter: Best K: "+str(best_cknn_k_abalone) + " / Best Sigma: "+str(best_cknn_sigma_abalone)+" / Best Epsilon: "+str(best_cknn_epsilon_abalone))

# Best KNN parameter:  Best K: 5 / Best Sigma: 2
# Best Edited KNN parameter: Best K: 10 / Best Sigma: 0.1 / Best Epsilon: 1
# Best Condensed KNN parameter: Best K: 8 / Best Sigma: 2 / Best Epsilon: 0.05

#Since it takes too long to do tuning process(2 and half hour), I just use parameter that I generated before.
best_knn_k_abalone, best_knn_sigma_abalone = 5,2
best_eknn_k_abalone, best_eknn_sigma_abalone, best_eknn_epsilon_abalone = 10, 0.1, 1
best_cknn_k_abalone, best_cknn_sigma_abalone, best_cknn_epsilon_abalone = 8, 2, 0.05

print("Best KNN parameter:  Best K: "+str(best_knn_k_abalone) + " / Best Sigma: "+str(best_knn_sigma_abalone))
print("Best Edited KNN parameter: Best K: "+str(best_eknn_k_abalone) + " / Best Sigma: "+str(best_eknn_sigma_abalone)+" / Best Epsilon: "+str(best_eknn_epsilon_abalone))
print("Best Condensed KNN parameter: Best K: "+str(best_cknn_k_abalone) + " / Best Sigma: "+str(best_cknn_sigma_abalone)+" / Best Epsilon: "+str(best_cknn_epsilon_abalone))

knn_score_abalone = knn_regression(validated_train_abalone, 'rings', best_knn_k_abalone, best_knn_sigma_abalone, False, "MSE", 'knn')[0]
knn_avg_score_abalone = round(np.mean(knn_score_abalone),4)
print("MSE for KNN: " + str(knn_score_abalone) + "/ Avg Score: "+ str(knn_avg_score_abalone))

eknn_score_abalone, eknn_remaining_abalone = knn_regression(validated_train_abalone, 'rings', best_eknn_k_abalone, best_eknn_sigma_abalone, best_eknn_epsilon_abalone, "MSE", 'edited knn')[:2]
eknn_avg_score_abalone= round(np.mean(eknn_score_abalone),4)
print("Remaining Data % after Edited KNN: "+ str(eknn_remaining_abalone)+ " / Avg Remaining Data %: "+ str(round(np.mean(eknn_remaining_abalone),2)))
print("MSE for Edited KNN: " + str(eknn_score_abalone) + " / Avg Score: "+ str(eknn_avg_score_abalone))

cknn_score_abalone, cknn_remaining_abalone = knn_regression(validated_train_abalone, 'rings', best_cknn_k_abalone, best_cknn_sigma_abalone, best_cknn_epsilon_abalone, "MSE", 'condensed knn')[:2]
cknn_avg_score_abalone = round(np.mean(cknn_score_abalone),4)
print("Remaining Data % after Condensed KNN: "+ str(cknn_remaining_abalone)+ " / Avg Remaining Data %: "+ str(round(np.mean(cknn_remaining_abalone),2)))
print("MSE for Condensed KNN: " + str(cknn_score_abalone) + " / Avg Score: "+ str(cknn_avg_score_abalone))

"""## Computer Hardware"""
print("-------Computer Hardware-------")
computer_dataset = open_csv_dataset('machine', True)
computer_dataset.isnull().sum().any()
computer_ERP  = computer_dataset['ERP']
computer_dataset = computer_dataset.drop(['vendor', 'model', 'ERP'], axis = 1)
computer_dataset_v1  = computer_dataset.copy()
computer_dataset_v1['PRP'] =computer_dataset_v1['PRP'].apply(computer_categorical_distribution)
computer_discretize_dataset = computer_dataset_v1.copy()
computer_discretize_dataset = discretization(computer_discretize_dataset, 
                                             5, 
                                             'equal_frequency', 
                                             ['myct','mmin', 'mmax', 'cach', 'chmin', 'chmax'])

computer_dataset_v2 = computer_discretize_dataset.copy()
train_computer_dataset, test_computer_dataset = split_dataset(computer_dataset_v2, 0.8)
train_computer_zscore_dataset, test_computer_zscore_dataset  = Standardization(train_computer_dataset,
                                                                               test_computer_dataset)

validated_train_computer = cross_validation_regression(train_computer_dataset, 5, 'PRP')
validated_train_computer_size = []
for i in range(0, 5):
    validated_train_computer_size.append(validated_train_computer[i].shape[0])
print("Size of Cross Validation: " + str(validated_train_computer_size))

tuning_computer = cross_validation_regression(test_computer_dataset , 5, 'PRP')
tuning_computer_size = []
for i in range(0, 5):
    tuning_computer_size.append(tuning_computer[i].shape[0])
print("Size of tuning: " + str(tuning_computer_size))

k_values = [i for i in range(1, 11)]
sigma_values =[0.01, 0.05, 0.1, 0.5, 1, 2]
epsilon_values = [0.01, 0.05, 0.1, 0.5, 1]

best_knn_k_computer, best_knn_sigma_computer = tuning_parameter_with_sigma(tuning_computer, k_values, 
                                                                           sigma_values, False, 
                                                                           'PRP', 'MSE', 'knn')
best_eknn_k_computer, best_eknn_sigma_computer, best_eknn_epsilon_computer= tuning_parameter_with_epsilon(tuning_computer, k_values, sigma_values,epsilon_values, 'PRP', 'MSE', 'edited knn')
best_cknn_k_computer, best_cknn_sigma_computer, best_cknn_epsilon_computer = tuning_parameter_with_epsilon(tuning_computer, k_values, sigma_values,epsilon_values, 'PRP', 'MSE', 'condensed knn')

print("Best KNN parameter:  Best K: "+str(best_knn_k_computer) + " / Best Sigma: "+str(best_knn_sigma_computer))
print("Best Edited KNN parameter: Best K: "+str(best_eknn_k_computer) + " / Best Sigma: "+str(best_eknn_sigma_computer)+" / Best Epsilon: "+str(best_eknn_epsilon_computer))
print("Best Condensed KNN parameter: Best K: "+str(best_cknn_k_computer) + " / Best Sigma: "+str(best_cknn_sigma_computer)+" / Best Epsilon: "+str(best_cknn_epsilon_computer))

knn_regress_computer = knn_demonstration_regression(validated_train_computer, 'PRP', best_knn_k_computer, best_knn_sigma_computer)

knn_regress_computer.head().style.set_properties(subset=['Assigned Class'], **{'width-min': '300px'})

knn_score_computer = knn_regression(validated_train_computer, 'PRP', best_knn_k_computer, best_knn_sigma_computer, False, "MSE", 'knn')[0]
knn_avg_score_computer = round(np.mean(knn_score_computer),4)
print("MSE for KNN: " + str(knn_score_computer) + "/ Avg Score: "+ str(knn_avg_score_computer))

eknn_score_computer, eknn_remaining_computer = knn_regression(validated_train_computer, 'PRP', best_eknn_k_computer, best_eknn_sigma_computer, best_eknn_epsilon_computer, "MSE", 'edited knn')[:2]
eknn_avg_score_computer= round(np.mean(eknn_score_computer),4)
print("Remaining Data % after Edited KNN: "+ str(eknn_remaining_computer) + " / Avg Remaining Data %: "+ str(round(np.mean(eknn_remaining_computer),2)))
print("MSE for Edited KNN: " + str(eknn_score_computer) + " / Avg Score: "+ str(eknn_avg_score_computer))

cknn_score_computer, cknn_remaining_computer = knn_regression(validated_train_computer, 'PRP', best_cknn_k_computer, best_cknn_sigma_computer, best_cknn_epsilon_computer, "MSE", 'condensed knn')[:2]
cknn_avg_score_computer = round(np.mean(cknn_score_computer),4)
print("Remaining Data % after Condensed KNN: "+ str(cknn_remaining_computer)+ " / Avg Remaining Data %: "+ str(round(np.mean(cknn_remaining_computer),2)))
print("MSE for Condensed KNN: " + str(cknn_score_computer) + " / Avg Score: "+ str(cknn_avg_score_computer))

"""## Forest Fires"""
print("-------Forest Fires-------")
forest_dataset = open_csv_dataset('forest', False)

forest_dataset.isnull().sum().any()
categorized_forest_dataset = replacing_string_to_numeric_multiple_columns(forest_dataset,['month', 'day'])
forest_dataset_v1  = categorized_forest_dataset.copy()
forest_dataset_v1['area'] = log_transform(forest_dataset_v1['area'])
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
print("Size of Cross Validation: " + str(validated_train_forest_size))

tuning_forest = cross_validation_regression(test_forest_dataset, 5, 'area')
tuning_forest_size  = []
for i in range(0, 5):
    tuning_forest_size .append(tuning_forest[i].shape[0])
print("Size of tuning: " + str(tuning_forest_size))

k_values = [i for i in range(1, 11)]
sigma_values =[0.01, 0.05, 0.1, 0.5, 1, 2]
epsilon_values = [0.01, 0.05, 0.1, 0.5, 1]

best_knn_k_forest, best_knn_sigma_forest = tuning_parameter_with_sigma(tuning_forest, k_values, 
                                                                           sigma_values, False, 
                                                                           'area', 'MSE', 'knn')
best_eknn_k_forest, best_eknn_sigma_forest, best_eknn_epsilon_forest= tuning_parameter_with_epsilon(tuning_forest, k_values, sigma_values,epsilon_values, 'area', 'MSE', 'edited knn')
best_cknn_k_forest, best_cknn_sigma_forest, best_cknn_epsilon_forest = tuning_parameter_with_epsilon(tuning_forest, k_values, sigma_values,epsilon_values, 'area', 'MSE', 'condensed knn')

print("Best KNN parameter:  Best K: "+str(best_knn_k_forest) + " / Best Sigma: "+str(best_knn_sigma_forest))
print("Best Edited KNN parameter: Best K: "+str(best_eknn_k_forest) + " / Best Sigma: "+str(best_eknn_sigma_forest)+" / Best Epsilon: "+str(best_eknn_epsilon_forest))
print("Best Condensed KNN parameter: Best K: "+str(best_cknn_k_forest) + " / Best Sigma: "+str(best_cknn_sigma_forest)+" / Best Epsilon: "+str(best_cknn_epsilon_forest))

knn_regress_forest = knn_demonstration_regression(validated_train_forest, 'area', best_knn_k_forest, best_knn_sigma_forest)

knn_regress_forest.head().style.set_properties(subset=['Assigned Class'], **{'width-min': '300px'})

knn_score_forest = knn_regression(validated_train_forest, 'area', best_knn_k_forest, best_knn_sigma_forest, False, "MSE", 'knn')[0]
knn_avg_score_forest = round(np.mean(knn_score_forest),4)
print("MSE for KNN: " + str(knn_score_forest) + "/ Avg Score: "+ str(knn_avg_score_forest))

eknn_score_forest, eknn_remaining_forest = knn_regression(validated_train_forest, 'area', best_eknn_k_forest, best_eknn_sigma_forest, best_eknn_epsilon_forest, "MSE", 'edited knn')[:2]
eknn_avg_score_forest= round(np.mean(eknn_score_forest),4)
print("Remaining Data % after Edited KNN: "+ str(eknn_remaining_forest)+ " / Avg Remaining Data %: "+ str(round(np.mean(eknn_remaining_forest),2)))
print("MSE for Edited KNN: " + str(eknn_score_forest) + " / Avg Score: "+ str(eknn_avg_score_forest))

cknn_score_forest, cknn_remaining_forest = knn_regression(validated_train_forest, 'area', best_cknn_k_forest, best_cknn_sigma_forest, best_cknn_epsilon_forest, "MSE", 'condensed knn')[:2]
cknn_avg_score_forest = round(np.mean(cknn_score_forest),4)
print("Remaining Data % after Condensed KNN: "+ str(cknn_remaining_forest)+ " / Avg Remaining Data %: "+ str(round(np.mean(cknn_remaining_forest),2)))
print("MSE for Condensed KNN: " + str(cknn_score_forest) + " / Avg Score: "+ str(cknn_avg_score_forest))