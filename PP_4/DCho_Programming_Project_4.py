#!/usr/bin/env python
# coding: utf-8
#DCho_Programing_Project_4.ipynb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import os
import glob
import math
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


# ## Sigmoid Function
#sigmoid function
#S(x)=1/(1+e^-x)
def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# ## Normalization
#Normalization
# It normalizes feature data in train dataset and test dataset.
# It doesn''t normalize predictor values
#x_norm = (x-min(x)) / (max(x)-min(x))
def normalize(train, test, predictor):
    new_train = train.copy()
    new_test = test.copy()
    new_train = new_train.loc[:, new_train.columns!=predictor]
    new_test = new_test.loc[:, new_test.columns!=predictor]
    train_min = new_train.min()
    train_max = new_train.max()
    normalized_train = np.divide(np.subtract(new_train, new_train.min()), np.subtract(new_train.max(), new_train.min()))
    normalized_test = np.divide(np.subtract(new_test, new_train.min()), np.subtract(new_train.max(), new_train.min()))
    normalized_train[predictor] = train[predictor]
    normalized_test[predictor] = test[predictor]
    return normalized_train, normalized_test


# ## Logistic Regression(LogR) for Binary Classes
# logistic Regression for Binary Classes (Based on Pseudocode Figure 10.6 in Introduction to machine learning Textbook)
# 1. Initializes weight values rangeing from -0.01 to 0.01
# 2. For every feature dataset, Calcualte the weight sum and pass weight sum to sigmoid function
# 3. Update the weight changed vector with gradient descent value returned from cost function. 
# 4. Multiply each weight change vector with learning rate
# 5. It repeats the process until it reaches to the maximum number of iterations. 
def LogR_GD_binary(train, class_df, max_iterations, learning_rate):
    iter_count = 0
    feature = len(train.columns)
    classes = len(class_df.columns)
    w_j = np.random.uniform(-0.01, 0.01, size=(classes, feature))
    while iter_count < max_iterations:
        delta_w_j = np.zeros(shape=(classes, feature))
        for i in range(train.shape[0]):
            o = 0 
            y = []
            for j in range(classes):
                o = o + np.dot(w_j[j], train.T[i])
                y = sigmoid(o)
                delta_w_j[j] = delta_w_j[j] + (class_df.loc[i].values[j]-y)*train.T[i]
        for j in range(classes):
            w_j[j] = w_j[j] + learning_rate * delta_w_j[j]      
        iter_count += 1
    return w_j

# Predict Logistic Regression for binary classes
# 1. It has final weight from logistic regression.
# 2. Using final weight from logistic regression, it comptues the y^t using sigmoid fuction.
# 3. If y^t is greater than equal to 0.5, it returns the Class 1. And Class 2 otherwise.
# 4. The class that has largest value is the most likely class for that instance. 
def predict_LogR_binary(weights, classes, test):
    pred = []
    for i in range(test.shape[0]):
        pred_y = []
        for j in range(len(classes)):
            final_w = np.dot(weights[j], test.iloc[i])
            sig_test = sigmoid(final_w)
            if sig_test >=0.5:
                predicted_val = 1
            else:
                predicted_val = 0
            pred_y.append(predicted_val)
        pred.append(classes[np.argmax(pred_y)])        
    return pred

# Evalute Logistic Regression binary
# 1. It normalizes the train and test dataset.
# 2. It gets class dataframe by converting categorical variable into indicator variables.
# 3. It gets final weight using logistic regression for binary class (LogR_GD_binary)
# 4. It predicts the output.
# 5. It evaluated the performance by computing accuracy score.
def LogR_binary_eval(df, test_df, predictor, max_iter, learning_rate, method):
    normalized_train, normalized_test  = normalize(df, test_df, predictor)
    class_df = pd.get_dummies(df[predictor])
    new_train_df = normalized_train[normalized_train.columns[normalized_train.columns!=predictor]]
    weight_vector = LogR_GD_binary(new_train_df, class_df, max_iter, learning_rate)
    new_test_df = test_df.drop(predictor, axis=1)
    prediction = predict_LogR_binary(weight_vector , class_df.columns, new_test_df)
    score = evaluation_metrics(test_df[predictor].values, prediction, method)
    score = round(score *100,2)
    return score

# Logistic Regresion binary for 5-fold
# It uses 80% of the dataset for 5-fold cross validation, and gets the accuracy score for each fold.
def LogR_binary_fold(df, predictor, max_iter, learning_rate, method):
    score_list = []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        k_validate = k_validate.reset_index(drop=True)
        score = LogR_binary_eval(k_validate, df[i], predictor, max_iter, learning_rate, method)
        score_list.append(score)
    return score_list

# ### Tuning Logistic Regression Binary Classes
# Tuning Logistic Regression Binary classe
# It uses 20% of the dataset for tuning. 
# It finds the best parameter combination of maximum iteration, learning rate. 
def tuning_LogR_single(df, max_iter_list, learning_rate_list, predictor, method):
    best_candidate = []
    parameter_list =[]
    for i in range(len(max_iter_list)):
        for j in range(len(learning_rate_list)):
            parameter_list.append([i,j])
            best_candidate.append(np.mean(LogR_binary_fold(df, 
                                                           predictor, 
                                                           max_iter_list[i], 
                                                           learning_rate_list[j],
                                                           method)))
    best_parameter =best_candidate.index(max(best_candidate))
    best_max_iter = max_iter_list[parameter_list[best_parameter][0]]
    best_learning_rate = learning_rate_list[parameter_list[best_parameter][1]]
    return best_learning_rate, best_max_iter

# ## Logistic Regresssion for Multi-Classes
# Logistic Regression for multiple class (Based on Pseudocode Figure 10.8 in Introduction to machine learning Textbook)
# 1. Initializes weight values rangeing from -0.01 to 0.01
# 2. For every feature dataset, Calcualte the weight sum.
#    Intead of using sigmoid function, it uses softmax function for multi-class
# 3. Just like logistic Regression for binary class
#    Update the weight changed vector with gradient descent value returned from cost function. 
# 4. Multiply each weight change vector with learning rate
# 5. It repeats the process until it reaches to the maximum number of iterations. 
def LogR_GD_multi_class(train, class_df, max_iterations, learning_rate):
    iter_count = 0
    feature = len(train.columns)
    classes = len(class_df.columns)
    w_j = np.random.uniform(-0.01, 0.01, size=(classes, feature))
    o = 0
    while(iter_count < max_iterations):
        delta_w_j = np.zeros(shape=(classes, feature))
        for i in range(train.shape[0]):
            o_i = []
            for j in range(classes):
                o = o + np.dot(w_j[j], train.T[i])
                o_i.append(o)
            y = []
            for j in range(classes):
                y.append(np.exp(o_i[j]) / np.sum(np.exp(o_i)))
            for j in range(classes):
                delta_w_j[j] = delta_w_j[j] + (class_df.loc[i].values[j]-y[j])*train.T[i]
        for j in range(classes):
            w_j[j] = w_j[j] + learning_rate * delta_w_j[j]
        iter_count += 1
    return w_j

# Prediction Logistic Regression for multi-class
# 1. Using logisitc regression for multi-class, it gets final weight
# 2. Instead of sigmoid function, it uses softmax. 
# 3. The class that has largest value is the most likely class for that instance. 
def predict_LogR_multi(weights, classes, test):
    pred = []
    for i in range(test.shape[0]):
        final_weight = []
        for j in range(len(classes)):
            final_weight.append(np.dot(weights[j], test.iloc[i]))
        y_i = []
        for j in range(len(classes)):
            y_i.append(np.exp(final_weight[j]) / np.sum(np.exp(final_weight)))
        pred.append(classes[np.argmax(y_i)])        
    return pred

# Evalute Logistic Regression multi-classes
# 1. It normalizes the train and test dataset.
# 2. It gets class dataframe by converting categorical variable into indicator variables.
# 3. It gets final weight using logistic regression for multi-class (LogR_GD_multi_class)
# 4. It predicts the output.
# 5. It evaluated the performance by computing accuracy score.
def LogR_multi_eval(df, test_df, predictor, max_iter, learning_rate, method):
    normalized_train, normalized_test  = normalize(df, test_df, predictor)
    class_df = pd.get_dummies(df[predictor])
    new_train_df = normalized_train[normalized_train.columns[normalized_train.columns!=predictor]]
    weight_vector =  LogR_GD_multi_class(new_train_df, class_df, max_iter, learning_rate)
    new_test_df = test_df.drop(predictor, axis=1)
    prediction = predict_LogR_multi(weight_vector , class_df.columns, new_test_df)
    score = evaluation_metrics(test_df[predictor].values, prediction, method)
    score = round(score *100,2)
    return score

# Logistic Regresion multi-class for 5-fold
# It uses 80% of the dataset for 5-fold cross validation, and gets the accuracy score for each fold.
def LogR_multi_fold(df, predictor, max_iter, learning_rate, method):
    score_list = []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        k_validate = k_validate.reset_index(drop=True)
        score = LogR_multi_eval(k_validate, df[i] ,predictor, max_iter, learning_rate, method)
        score_list.append(score)
    return score_list


# ### Tuning Logistic Regression for Multi-Classes
# Tuning Logistic Regression Multi-classes
# It uses 20% of the dataset for tuning. 
# It finds the best parameter combination of maximum iteration, learning rate. 
def tuning_LogR_multi(df, max_iter_list, learning_rate_list, predictor, method):
    best_candidate = []
    parameter_list =[]
    for i in range(len(max_iter_list)):
        for j in range(len(learning_rate_list)):
            parameter_list.append([i,j])
            best_candidate.append(np.mean(LogR_multi_fold(df, 
                                                          predictor, 
                                                          max_iter_list[i], 
                                                          learning_rate_list[j],
                                                          method)))
    best_parameter =best_candidate.index(max(best_candidate))
    best_max_iter = max_iter_list[parameter_list[best_parameter][0]]
    best_learning_rate = learning_rate_list[parameter_list[best_parameter][1]]
    return best_learning_rate, best_max_iter


# ## Simple Linear Network / Linear Regression (LinR)
# Simple Liner Network (Linear Regression for regression tasks)
# 1. Initializes weight values rangeing from -0.01 to 0.01
# 2. It computes the weight sum for every feature dataset. 
# 3. It updates the weight changed vector with gradient descent value using MSE.
# 4. Multiply each weight change vector with learning rate
# 5. It repeats the process until it reaches to the maximum number of iterations. 
def LinR_GD(train, predictor, learning_rate):
    features = train.columns[train.columns!=predictor]
    new_train = train[features]
    w_j = np.random.uniform(-0.01, 0.01, len(features))
    delta_w_ij = np.zeros(shape=(1, len(features)))
    for i in range(new_train.shape[0]):
        pred_o = np.dot(w_j,new_train.iloc[i].values)
        delta_w_ij[0] = delta_w_ij[0] + np.square(train[predictor].values[i]-pred_o)*new_train.iloc[i]
    MSE = (1/new_train.shape[0])*delta_w_ij
    w_j = w_j + learning_rate * MSE
    return w_j

# Prediction Linear Regression
# 1. For every feature dataset, it computes the weight vector to predict the result.
def predict_LinR(df, weight, predictor):
    df = df.iloc[:, df.columns!=predictor]
    pred = []
    for i in range(df.shape[0]):
        pred.append(np.dot(weight,df.iloc[i].values)[0])
    return pred

# Linear Regression Evaluation
# 1. It normalizes the dataset for trian and test dataset. 
# 2. It gets the final weight for linear regression. 
# 3. It predicts the result.
# 4. It evaluted the prediction using MSE (Mean squared error)
def LinR_eval(df, test_df, predictor, learning_rate, method):
    normalized_df, normalized_test_df  = normalize(df, test_df, predictor)
    weight = LinR_GD(normalized_df, predictor, learning_rate)
    prediction = predict_LinR(normalized_test_df, weight, predictor)
    score = evaluation_metrics(normalized_test_df[predictor].values, prediction, method)
    score = round(score,2)
    return score

# Linear Regression for 5-fold
# It uses 80% of the dataset for 5-fold cross validation, and gets the MSE for each fold.
def LinR_fold(df, predictor, learning_rate, method):
    score_list = []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        k_validate = k_validate.reset_index(drop=True)
        score = LinR_eval(k_validate, df[i], predictor ,learning_rate, method)
        score_list.append(score)
    return score_list

# Tuning Linear Regression Multi-classes
# It uses 20% of the dataset for tuning. 
# It finds the best parameter combination of maximum iteration, learning rate. 
def tuning_LinR(df, learning_rate_list, predictor, method):
    best_candidate = []
    parameter_list = []
    for i in range(len(learning_rate_list)):
        parameter_list.append(i)
        best_candidate.append(np.mean(LinR_fold(df, predictor, learning_rate_list[i], method)))
    best_parameter = best_candidate.index(min(best_candidate))
    best_learning_rate = learning_rate_list[parameter_list[best_parameter]]
    return best_learning_rate


# ## FeedForward with Backpropagation (FFBP)
# ## FFBP for Classification
# Backpropagation  (Based on Pseudocode Figure 11.11 in Introduction to machine learning Textbook)
# 1. There are two hidden layer for this project
# 2. Initialize the weight ranges from -0.01 to 0.01 for 
#    (Input to first hidden layer), (first hidden layer to second hidden layer), (second hidden layer to output layer)
# 3. For each class, Compute weight update rule for second layer
#    deltav_h = eta * sum_t(r^t-y^t)*z^t_h
# 4. For each class, Compute weight update rule for first layer
#    deltaw_hj = eta * sum_t(r^t-y^t)v_h*z^t_h(1-z^t_h)x^t_j
# 5. Since it has two hidden layer for this project, it repeats weight upating the same process again.
# 6. It repeats the whole process until it reaches to max iteration numbers. 
def BP_classification(df, class_df, hidden_layer1, hidden_layer2, max_iter, learning_rate):
    feature = len(df.columns)
    classes = len(class_df.columns)
    w_j = np.random.uniform(-0.01, 0.01, size=(hidden_layer1, feature))
    v_ih1 = np.random.uniform(-0.01, 0.01, size=(hidden_layer2, hidden_layer1))
    v_ih2 = np.random.uniform(-0.01, 0.01, size=(classes, hidden_layer2))
    iteration = 0
    while iteration < max_iter:
        for i in range(df.shape[0]):
            #First Layer
            z_h1 = []
            for h1 in range(hidden_layer1):
                z_h1.append(sigmoid(np.dot(w_j[h1], df.T[i])))
            y_i_h1=[]
            for j in range(classes):
                y_i_h1.append(np.dot(v_ih1[j], z_h1))
            delta_vi_h1 = np.zeros(shape=(classes, hidden_layer1))
            for j in range(classes):
                gd_h1 = learning_rate*(class_df.loc[i].values[j] - y_i_h1[j]) 
                gd_v_h1 = [gd_h1* x for x in z_h1]
                delta_vi_h1[j]=  delta_vi_h1[j] + gd_v_h1
            delta_w_h1 = np.zeros(shape=(hidden_layer1, feature))
            for h1 in range(hidden_layer1):
                weight_sum = []
                for j in range(classes):
                    weight_sum.append((class_df.loc[i].values[j] - y_i_h1[j])*v_ih1[j][h1])
                delta_w_h1[h1] = delta_w_h1[h1] + learning_rate*np.sum(weight_sum)*z_h1[h1]*(1-z_h1[h1])* df.T[i]
            for j in range(classes):
                v_ih1[j] = v_ih1[j] + delta_vi_h1[j]
            for h1 in range(hidden_layer1):
                w_j[h1] = w_j[h1]+delta_w_h1[h1]
            #Second Layer
            z_h2 = []
            for h2 in range(hidden_layer2):
                z_h2.append(sigmoid(np.dot(v_ih1[h2], z_h1)))
            y_i_h2=[]
            for j in range(classes):
                y_i_h2.append(np.dot(v_ih2[j], z_h2))
            delta_vi_h2 = np.zeros(shape=(classes, hidden_layer2))
            for j in range(classes):
                gd_h2 = learning_rate*(class_df.loc[i].values[j] - y_i_h2[j]) 
                gd_v_h2 = [gd_h2* x for x in z_h2]
                delta_vi_h2[j]=  delta_vi_h2[j] + gd_v_h2
            delta_w_h2 = np.zeros(shape=(hidden_layer2, hidden_layer1))
            for h2 in range(hidden_layer2):
                weight_sum = []
                for j in range(classes):
                    weight_sum.append((class_df.loc[i].values[j] - y_i_h2[j])*v_ih2[j][h2])
                w_h2 = learning_rate*np.sum(weight_sum)*z_h2[h2]*(1-z_h2[h2])
                gd_w_h2 = [w_h2* x for x in z_h1]
                delta_w_h2[h2] = delta_w_h2[h2] + gd_w_h2
            for j in range(classes):
                v_ih2[j] = v_ih2[j] + delta_vi_h2[j]
            for h2 in range(hidden_layer2):
                v_ih1[h2] = v_ih1[h2]+delta_w_h2[h2]
        iteration+=1
    return w_j, v_ih1, v_ih2

# Predict Feedforward with Backpropagation foor classification
# It predicts the classification using initial weight, first weight, second weight from BP_classification.
# It computes the final weight vector using sigmoid for each layer.
# In outputer layer, it predicts the final class value using softmax. 
# The class that has largest value is the most likely class for that instance. 
def predict_FFBP_classification(df, class_df, h1_node, h2_node, weight, h1_layer, h2_layer):
    df_test = df.reset_index(drop=True)
    pred = []
    for i in range(df_test.shape[0]):
        z_h1 = []
        # First Layer 
        for h1 in range(h1_node):
            z_h1.append(sigmoid(np.dot(weight[h1], df_test.iloc[i])))
        z_h2 = []
        #Second Layer
        for h2 in range(h2_node):
            z_h2.append(sigmoid(np.dot(h1_layer[h2], z_h1)))
        #Output Layer
        final_weight = []
        for j in range(len(class_df)):
            final_weight.append(np.dot(h2_layer[j], z_h2))
        #SoftMax for prediction Layer
        soft_max_pred = []
        for j in range(len(class_df)):
            soft_max_pred.append(np.exp(final_weight[j]) / np.sum(np.exp(final_weight)))
        pred.append(class_df[np.argmax(soft_max_pred)])
    return pred

#Feedforward backpropgation with clssification
# 1. It normalizes the dataset for trian and test dataset. 
# 2. It gets the inital weight, weight of first hidden layer, weight for second hidden layer from BP_classification
# 3. It predicts the result.
# 4. It evaluted the prediction using accuracy score.
def FFBP_classification(df, test_df, predictor, h1_node, h2_node, max_iter, learning_rate, method):
    normalized_train, normalized_test  = normalize(df, test_df, predictor)
    class_df = pd.get_dummies(df[predictor])
    new_train_df = normalized_train[normalized_train.columns[normalized_train.columns!=predictor]]
    weight, hidden_layer1, hidden_layer2 = BP_classification(new_train_df, class_df, h1_node, h2_node, max_iter, learning_rate)
    new_test_df = normalized_test.drop(predictor, axis=1)
    prediction = predict_FFBP_classification(new_test_df, class_df.columns, h1_node, h2_node, weight, hidden_layer1, hidden_layer2)
    score = evaluation_metrics(test_df[predictor].values, prediction, method)
    score = round(score*100,2)
    return score

# Feedforward backpropagation for 5-fold cross validation
# It uses 80% of the dataset for 5-fold cross validation, and gets the accuracy score for each fold.
def FFBP_classification_fold(df, predictor, h1_node, h2_node,  max_iter, learning_rate, method):
    score_list = []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        k_validate = k_validate.reset_index(drop=True)
        score = FFBP_classification(k_validate, df[i], predictor, h1_node, h2_node, max_iter, learning_rate, method)
        score_list.append(score)
    return score_list

# ### Tuning FFBP Classification 
# Tuning Feedforward backpropagation with classification
# It uses 20% of the dataset for tuning. 
# It finds the best parameter combination of maximum iteration, learning rate, hidden node 
def tuning_FFBP_classification(df, max_iter_list, learning_rate_list, hidden_node_list, predictor, method):
    best_candidate = []
    parameter_list =[]
    for i in range(len(max_iter_list)):
        for j in range(len(learning_rate_list)):
            for k in range(len(hidden_node_list)):
                parameter_list.append([i,j, k])
                best_candidate.append(np.mean(FFBP_classification_fold(df, 
                                                                       predictor, 
                                                                       hidden_node_list[k], 
                                                                       hidden_node_list[k], 
                                                                       max_iter_list[i], 
                                                                       learning_rate_list[j], 
                                                                       method)))     
    best_parameter =best_candidate.index(max(best_candidate))
    best_max_iter = max_iter_list[parameter_list[best_parameter][0]]
    best_learning_rate = learning_rate_list[parameter_list[best_parameter][1]]
    best_hidden_node = hidden_node_list[parameter_list[best_parameter][2]]
    return best_learning_rate, best_max_iter, best_hidden_node


# ## FFBP for Regression
# Feedforward backpropagation for regression tasks.
# It basically works same as Feedforward backpropagation for classification tasks.
# Except that regression tasks don't have to care about the classes label. 
# 1. There are two hidden layer for this project
# 2. Initialize the weight for 
#    (Input to first hidden layer), (first hidden layer to second hidden layer), (second hidden layer to output layer)
# 3. Compute weight update rule for second layer
#    deltav_h = eta * sum_t(r^t-y^t)*z^t_h
# 4. Compute weight update rule for first layer
#    deltaw_hj = eta * sum_t(r^t-y^t)v_h*z^t_h(1-z^t_h)x^t_j
# 5. Since it has two hidden layer for this project, it repeats weight upating the same process again.
# 6. It repeats the whole process until it reaches to max iteration numbers. 
def BP_regression(df, class_df, hidden_layer1, hidden_layer2, max_iter, learning_rate):
    feature = len(df.columns)
    weight = np.random.uniform(-0.01, 0.01, size=(hidden_layer1, feature))
    h1_layer = np.random.uniform(-0.01, 0.01, size=(hidden_layer2, hidden_layer1))
    h2_layer = np.random.uniform(-0.01, 0.01, size=(1, hidden_layer2))
    iteration = 0
    while iteration < max_iter:
        for i in range(df.shape[0]):
            #First layer
            z_h1 = []
            for h1 in range(hidden_layer1):
                z_h1.append(sigmoid(np.dot(weight[h1],df.T[i])))
            y_i_h1 = (np.dot(h1_layer, z_h1))
            delta_v_i_1 = learning_rate *(class_df[i]- y_i_h1)*z_h1
            delta_wh_1 = np.zeros(shape=(hidden_layer1, feature))
            for h1 in range(hidden_layer1):
                gd_weight_sum_h1 = (class_df[i] - y_i_h1)*delta_v_i_1[h1]
                delta_wh_1[h1] = delta_wh_1[h1] + learning_rate*np.sum(gd_weight_sum_h1)*z_h1[h1]*(1-z_h1[h1])* df.T[i]
            h1_layer  = h1_layer+delta_v_i_1
            for h1 in range(hidden_layer1):
                weight[h1] = weight[h1]+delta_wh_1[h1]
            #Second Layer 
            z_h2 = []
            for h2 in range(hidden_layer2):
                z_h2.append(sigmoid(np.dot(h1_layer[h2],z_h1)))
            y_i_h2 = (np.dot(h2_layer, z_h2))
            delta_v_i_2 = learning_rate *(class_df[i]- y_i_h2)*z_h2
            delta_wh_2 = np.zeros(shape=(hidden_layer2, hidden_layer1))
            for h2 in range(hidden_layer2):
                gd_w_h2 = (class_df[i] - y_i_h2)*delta_v_i_2[h2]
                weight_sum_h2 = learning_rate*np.sum(gd_w_h2)*z_h2[h2]*(1-z_h2[h2])
                weight_h2 = [weight_sum_h2* x for x in z_h1]
                delta_wh_2[h2] = delta_wh_2[h2] + weight_h2 
            h2_layer  = h2_layer+delta_v_i_2
            for h2 in range(hidden_layer2):
                h1_layer[h2] = h1_layer[h2]+delta_wh_2[h2]
        iteration+=1
    return weight, h1_layer, h2_layer

# Predict Feedforward with Backpropagation for regression
# It predicts the regression using initial weight, first weight, second weight from BP_regression.
# It computes the final weight vector using sigmoid for each layer.
# In outputer layer, it predicts the final value using softmax. 
def predict_FFBP_regression(df, weight, h1_node, h2_node, h1_layer, h2_layer):
    pred =[]
    for i in range(df.shape[0]):
        z_h1 = []
        # First Layer 
        for h1 in range(h1_node):
            z_h1.append(sigmoid(np.dot(weight[h1], df.iloc[i])))
        z_h2 = []
        #Second Layer
        for h2 in range(h2_node):
            z_h2.append(sigmoid(np.dot(h1_layer[h2], z_h1)))
        final_weight = np.dot(h2_layer, z_h2)
        pred.append(final_weight[0])
    return pred

#Feedforward backpropgation with regression
# 1. It normalizes the dataset for train and test dataset. 
# 2. It gets the inital weight, weight of first hidden layer, weight for second hidden layer from BP_regression
# 3. It predicts the result.
# 4. It evaluted the prediction using MSE.
def FFBP_regression(df, test_df, predictor, h1_node, h2_node, max_iter, learning_rate, method):
    normalized_train, normalized_test  = normalize(df, test_df, predictor)
    new_train_df = normalized_train[normalized_train.columns[normalized_train.columns!=predictor]]
    class_df = normalized_train[predictor].values
    weight, h1_layer, h2_layer = BP_regression(new_train_df, class_df, h1_node, h2_node, max_iter, learning_rate)
    normalized_test_df = normalized_test.reset_index(drop=True)
    new_test_df = normalized_test_df[normalized_test_df.columns[normalized_test_df.columns!=predictor]]
    prediction = predict_FFBP_regression(new_test_df, weight, h1_node, h2_node, h1_layer, h2_layer)
    score = evaluation_metrics(normalized_test_df[predictor].values, prediction, method)
    score = round(score,2)
    return score

# Feedforward backpropagation for 5-fold cross validation
# It uses 80% of the dataset for 5-fold cross validation, and gets the MSE for each fold.
def FFBP_regression_fold(df, predictor, h1_node, h2_node,  max_iter, learning_rate, method):
    score_list = []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        k_validate = k_validate.reset_index(drop=True)
        score = FFBP_regression(k_validate, df[i], predictor, h1_node, h2_node, max_iter, learning_rate, method)
        score_list.append(score)
    return score_list

# Tuning Feedforward backpropagation with regression
# It uses 20% of the dataset for tuning. 
# It finds the best parameter combination of maximum iteration, learning rate, hidden node 
def tuning_FFBP_regression(df, max_iter_list, learning_rate_list, hidden_node_list, predictor, method):
    best_candidate = []
    parameter_list =[]
    for i in range(len(max_iter_list)):
        for j in range(len(learning_rate_list)):
            for k in range(len(hidden_node_list)):
                parameter_list.append([i,j, k])
                best_candidate.append(np.mean(FFBP_regression_fold(df, 
                                                                   predictor, 
                                                                   hidden_node_list[k], 
                                                                   hidden_node_list[k], 
                                                                   max_iter_list[i], 
                                                                   learning_rate_list[j], 
                                                                   method)))     
    best_parameter =best_candidate.index(min(best_candidate))
    best_max_iter = max_iter_list[parameter_list[best_parameter][0]]
    best_learning_rate = learning_rate_list[parameter_list[best_parameter][1]]
    best_hidden_node = hidden_node_list[parameter_list[best_parameter][2]]
    return best_learning_rate, best_max_iter, best_hidden_node


# ## AutoEncoder
# ### Feature Extraction (Encoding Layer)
# Autoencoder for both classification and regression
# In Autoencoder, the number of encoding layer is smaller ther nodes in the input layer.
# It tries to reduce feature space that is projecting into.
# It uses feedforward with backpropagation to train the data in the encoding layer.
def autoencoder(df, encoding_layer1, max_iter, learning_rate):
    feature = len(df.columns)
    w_j = np.random.uniform(-0.01, 0.01, size=(encoding_layer1, feature))
    v_ih1 = np.random.uniform(-0.01, 0.01, size=(feature, encoding_layer1))
    iterations = 0
    while iterations < max_iter:
        z_weight =[]
        for i in range(df.shape[0]):
            #Input to Encoding Layer(Hidden Layer)
            z_h1 = []
            for h1 in range(encoding_layer1):
                z_h1.append(sigmoid(np.dot(w_j[h1], df.T[i])))
            z_weight.append(z_h1)
            y_i_h1=[]
            for j in range(feature):
                y_i_h1.append(np.dot(v_ih1[j], z_h1))
            delta_vi_h1 = np.zeros(shape=(feature, encoding_layer1))
            for j in range(feature):
                gd_h1 = learning_rate*(df.iloc[i][j] - y_i_h1[j]) 
                gd_v_h1 = [gd_h1* x for x in z_h1]
                delta_vi_h1[j]=  delta_vi_h1[j] + gd_v_h1
            delta_w_h1 = np.zeros(shape=(encoding_layer1, feature))
            for h1 in range(encoding_layer1):
                weight_sum = []
                for j in range(feature):
                    weight_sum.append((df.iloc[i][j] - y_i_h1[j])*v_ih1[j][h1])
                delta_w_h1[h1] = delta_w_h1[h1] + learning_rate*np.sum(weight_sum)*z_h1[h1]*(1-z_h1[h1])* df.T[i]
            for j in range(feature):
                v_ih1[j] = v_ih1[j] + delta_vi_h1[j]
            for h1 in range(encoding_layer1):
                w_j[h1] = w_j[h1]+delta_w_h1[h1]
        iterations += 1    
    return w_j, z_weight


# ### Classification
# Autoencoder classification 
# It uses weight from autoencoder function above. 
# It uses feedforward with backpropagation for hidden layer to output layer.
def autoencoder_classification(df, class_df,  hidden_layer, max_iter, learning_rate):
    classes = len(class_df.columns)
    w_j = np.random.uniform(-0.01, 0.01, size=(hidden_layer, len(df.columns)))
    v_ih = np.random.uniform(-0.01, 0.01, size=(classes, hidden_layer))
    iterations = 0
    while iterations < max_iter:
        for i in range(df.shape[0]):
            z_h = []
            for e1 in range(hidden_layer):
                z_h.append(sigmoid(np.dot(w_j[e1], df.T[i])))
            y_i_h=[]
            for j in range(classes):
                y_i_h.append(np.dot(v_ih[j], z_h))
            delta_vi_h = np.zeros(shape=(classes, hidden_layer))
            for j in range(classes):
                gd_h = learning_rate*(class_df.iloc[i].values[j] - y_i_h[j]) 
                gd_v_h = [gd_h* x for x in z_h]
                delta_vi_h[j]=  delta_vi_h[j] + gd_v_h
            delta_w_h = np.zeros(shape=(hidden_layer, len(df.columns)))
            for e1 in range(hidden_layer):
                weight_sum = []
                for j in range(classes):
                    weight_sum.append((class_df.iloc[i].values[j] - y_i_h[j])*v_ih[j][e1])
                delta_w_h[e1] = delta_w_h[e1] + learning_rate*np.sum(weight_sum)*z_h[e1]*(1-z_h[e1])* df.T[i]
            for j in range(classes):
                v_ih[j] = v_ih[j] + delta_vi_h[j]
            for e1 in range(hidden_layer):
                w_j[e1] = w_j[e1]+delta_w_h[e1]
        iterations+=1
    return w_j, v_ih

# Autoencoder Feedforward backpropagation
# After it constructs the network using autoencoder,
#  it trains with feedforward with backpropagation that has two hidden layer from above. 
# It takes initial weight, weight of encoding layer, weight of hidden layer for feedforward with backpropagation.
def BP_encoding_class(df, class_df, w_j, v_ih1, v_ih2,  max_iter, learning_rate):
    feature = len(df.columns)
    classes = len(class_df.columns)
    hidden_layer1 = len(w_j)
    hidden_layer2 = len(v_ih1)
    iteration = 0
    while iteration < max_iter:
        for i in range(df.shape[0]):
            #Input to First Hidden Layer
            z_h1 = []
            for h1 in range(hidden_layer1):
                z_h1.append(sigmoid(np.dot(w_j[h1], df.T[i])))
            y_i_h1=[]
            for j in range(classes):
                y_i_h1.append(np.dot(v_ih1[j], z_h1))
            delta_vi_h1 = np.zeros(shape=(classes, hidden_layer1))
            for j in range(classes):
                gd_h1 = learning_rate*(class_df.loc[i].values[j] - y_i_h1[j]) 
                gd_v_h1 = [gd_h1* x for x in z_h1]
                delta_vi_h1[j]=  delta_vi_h1[j] + gd_v_h1
            delta_w_h1 = np.zeros(shape=(hidden_layer1, feature))
            for h1 in range(hidden_layer1):
                weight_sum = []
                for j in range(classes):
                    weight_sum.append((class_df.loc[i].values[j] - y_i_h1[j])*v_ih1[j][h1])
                delta_w_h1[h1] = delta_w_h1[h1] + learning_rate*np.sum(weight_sum)*z_h1[h1]*(1-z_h1[h1])* df.T[i]
            for j in range(classes):
                v_ih1[j] = v_ih1[j] + delta_vi_h1[j]
            for h1 in range(hidden_layer1):
                w_j[h1] = w_j[h1]+delta_w_h1[h1]
            #First Hidden layer to Second hidden Layer
            z_h2 = []
            for h2 in range(hidden_layer2):
                z_h2.append(sigmoid(np.dot(v_ih1[h2], z_h1)))
            y_i_h2=[]
            for j in range(classes):
                y_i_h2.append(np.dot(v_ih2[j], z_h2))
            delta_vi_h2 = np.zeros(shape=(classes, hidden_layer2))
            for j in range(classes):
                gd_h2 = learning_rate*(class_df.loc[i].values[j] - y_i_h2[j]) 
                gd_v_h2 = [gd_h2* x for x in z_h2]
                delta_vi_h2[j]=  delta_vi_h2[j] + gd_v_h2
            delta_w_h2 = np.zeros(shape=(hidden_layer2, hidden_layer1))
            for h2 in range(hidden_layer2):
                weight_sum = []
                for j in range(classes):
                    weight_sum.append((class_df.loc[i].values[j] - y_i_h2[j])*v_ih2[j][h2])
                w_h2 = learning_rate*np.sum(weight_sum)*z_h2[h2]*(1-z_h2[h2])
                gd_w_h2 = [w_h2* x for x in z_h1]
                delta_w_h2[h2] = delta_w_h2[h2] + gd_w_h2
            for j in range(classes):
                v_ih2[j] = v_ih2[j] + delta_vi_h2[j]
            for h2 in range(hidden_layer2):
                v_ih1[h2] = v_ih1[h2]+delta_w_h2[h2]
        iteration+=1
    return w_j, v_ih1, v_ih2

# Evalute Autoencoder Feedforward with backpropagation for classification
# 1. It normalizes the train and test dataset.
# 2. It gets class dataframe by converting categorical variable into indicator variables.
# 3. It gets encoding weight and z_h weight using autoencoder.
# 4. It gets hidden weight and output weight using autoencoder classification.
# 5. Using encoding weight, hidden weight, and output weight, it trains network with backpropagation (BP_encoding_class)
# 4. It predicts the output.
# 5. It evaluated the performance by computing accuracy score.
def autoencoding_FFBP_classification(df, test_df, predictor, encoding_node, hidden_node, max_iter, learning_rate, method):
    normalized_train, normalized_test  = normalize(df, test_df, predictor)
    class_df = pd.get_dummies(df[predictor])
    new_train_df = normalized_train[normalized_train.columns[normalized_train.columns!=predictor]]
    encoding_weight, z_hweight= autoencoder(new_train_df, encoding_node, max_iter, learning_rate)
    z_hweight_df = pd.DataFrame(z_hweight)
    hidden_weight , output_weight = autoencoder_classification(z_hweight_df, class_df,  hidden_node, max_iter, learning_rate)
    new_test_df = normalized_test.drop(predictor, axis=1)
    weight, hidden_layer1, hidden_layer2 = BP_encoding_class(new_train_df, class_df, encoding_weight, hidden_weight, output_weight, max_iter, learning_rate)
    backprop_prediction = predict_FFBP_classification(new_test_df, class_df.columns, encoding_node, hidden_node, weight, hidden_layer1, hidden_layer2)
    score = evaluation_metrics(test_df[predictor].values, backprop_prediction, method)
    score = round(score*100,2)
    return score

# Autoencoder Feedforward backpropagation for 5-fold cross validation
# It uses 80% of the dataset for 5-fold cross validation, and gets the accuracy score for each fold.
def autoencoder_classification_fold(df, predictor, encoding_node, hidden_node,  max_iter, learning_rate, method):
    score_list = []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        k_validate = k_validate.reset_index(drop=True)
        score = autoencoding_FFBP_classification(k_validate, df[i], predictor, encoding_node, hidden_node, max_iter, learning_rate, method)
        score_list.append(score)
    return score_list

# Tuning Autoencoder Feedforward backpropagation for classification
# It uses 20% of the dataset for tuning. 
# It finds the best parameter combination of maximum iteration, learning rate, encoding node, hidden node.
def tuning_autoencoder_classification(df, max_iter_list, learning_rate_list, encoding_node_list, hidden_node_list, predictor, method):
    best_candidate = []
    parameter_list =[]
    for i in range(len(max_iter_list)):
        for j in range(len(learning_rate_list)):
            for k in range(len(encoding_node_list)):
                for m in range(len(hidden_node_list)):
                    parameter_list.append([i,j, k, m])
                    best_candidate.append(np.mean(autoencoder_classification_fold(df, 
                                                                               predictor, 
                                                                               encoding_node_list[k], 
                                                                               hidden_node_list[m], 
                                                                               max_iter_list[i], 
                                                                               learning_rate_list[j], 
                                                                               method)))     
    best_parameter =best_candidate.index(max(best_candidate))
    best_max_iter = max_iter_list[parameter_list[best_parameter][0]]
    best_learning_rate = learning_rate_list[parameter_list[best_parameter][1]]
    best_encoding_node =encoding_node_list[parameter_list[best_parameter][2]]
    best_hidden_node = hidden_node_list[parameter_list[best_parameter][3]]
    return best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node


# ### Regression
# Just like Autoencoder classification, Autoencoder_regression works almost same way. 
# It uses weight from autoencoder function above. 
# It uses feedforward with backpropagation for hidden layer to output layer.
def autoencoder_regression(df, class_df, hidden_layer, max_iter, learning_rate):
    w_j = np.random.uniform(-0.01, 0.01, size=(hidden_layer, len(df.columns)))
    v_ih  = np.random.uniform(-0.01, 0.01, size=(1, hidden_layer))
    iteration = 0
    while iteration < max_iter:
        for i in range(df.shape[0]):
            z_h1 = []
            for h1 in range(hidden_layer):
                z_h1.append(sigmoid(np.dot(w_j[h1],df.T[i])))
            y_i_h1 = (np.dot(v_ih, z_h1))
            delta_v_i_1 = learning_rate *(class_df[i]- y_i_h1)*z_h1
            delta_wh_1 = np.zeros(shape=(hidden_layer, len(df.columns)))
            for h1 in range(hidden_layer):
                gd_w_j_sum_h1 = (class_df[i] - y_i_h1)*delta_v_i_1[h1]
                delta_wh_1[h1] = delta_wh_1[h1] + learning_rate*np.sum(gd_w_j_sum_h1)*z_h1[h1]*(1-z_h1[h1])* df.T[i]
            v_ih  = v_ih+delta_v_i_1
            for h1 in range(hidden_layer):
                w_j[h1] = w_j[h1]+delta_wh_1[h1]
        iteration+=1
    return w_j, v_ih

# Autoencoder Feedforward backpropagation
# After it constructs the network using autoencoder,
#  it trains with feedforward with backpropagation that has two hidden layer from above. 
# It takes initial weight, weight of encoding layer, weight of hidden layer for feedforward with backpropagation.
def BP_encoding_regression(df, class_df, weight, h1_layer, h2_layer, max_iter, learning_rate):
    feature = len(df.columns)
    hidden_layer1 = len(weight)
    hidden_layer2 = len(h1_layer)
    iteration = 0
    while iteration < max_iter:
        for i in range(df.shape[0]):
            #First layer
            z_h1 = []
            for h1 in range(hidden_layer1):
                z_h1.append(sigmoid(np.dot(weight[h1],df.T[i])))
            y_i_h1 = (np.dot(h1_layer, z_h1))
            delta_v_i_1 = np.zeros(shape=(hidden_layer2, hidden_layer1))
            for h2 in range(hidden_layer2):
                gd_h1 = learning_rate*(class_df[i] - y_i_h1[h2]) 
                gd_v_h1 = [gd_h1* x for x in z_h1]
                delta_v_i_1[h2]=  delta_v_i_1[h2] + gd_v_h1
            delta_wh_1 = np.zeros(shape=(hidden_layer1, feature))
            for h1 in range(hidden_layer1):
                for h2 in range(hidden_layer2):
                    gd_weight_sum_h1 = (class_df[i] - y_i_h1)*delta_v_i_1[h2][h1]
                    delta_wh_1[h1] = delta_wh_1[h1] + learning_rate*np.sum(gd_weight_sum_h1)*z_h1[h1]*(1-z_h1[h1])* df.T[i]
            for h2 in range(hidden_layer2):
                h1_layer[h2]  = h1_layer[h2] +delta_v_i_1[h2]
            for h1 in range(hidden_layer1):
                weight[h1] = weight[h1]+delta_wh_1[h1]
            #Second Layer 
            z_h2 = []
            for h2 in range(hidden_layer2):
                z_h2.append(sigmoid(np.dot(h1_layer[h2],z_h1)))
            y_i_h2 = (np.dot(h2_layer, z_h2))
            delta_v_i_2 = learning_rate *(class_df[i]- y_i_h2)*z_h2
            delta_wh_2 = np.zeros(shape=(hidden_layer2, hidden_layer1))
            for h2 in range(hidden_layer2):
                gd_w_h2 = (class_df[i] - y_i_h2)*delta_v_i_2[h2]
                weight_sum_h2 = learning_rate*np.sum(gd_w_h2)*z_h2[h2]*(1-z_h2[h2])
                weight_h2 = [weight_sum_h2* x for x in z_h1]
                delta_wh_2[h2] = delta_wh_2[h2] + weight_h2 
            h2_layer  = h2_layer+delta_v_i_2
            for h2 in range(hidden_layer2):
                h1_layer[h2] = h1_layer[h2]+delta_wh_2[h2]
        iteration+=1
    return weight, h1_layer, h2_layer

# Evalute Autoencoder Feedforward with backpropagation for regression
# 1. It normalizes the train and test dataset.
# 2. It gets class dataframe by converting categorical variable into indicator variables.
# 3. It gets encoding weight and z_h weight using autoencoder.
# 4. It gets hidden weight and output weight using autoencoder regression.
# 5. Using encoding weight, hidden weight, and output weight, it trains network with backpropagation (BP_encoding_regression)
# 4. It predicts the output.
# 5. It evaluated the performance by computing accuracy score.
def autoencoding_FFBP_regression(df, test_df, predictor, encoding_node, hidden_node, max_iter, learning_rate, method):
    normalized_train, normalized_test  = normalize(df, test_df, predictor)
    class_df = normalized_train[predictor].values
    new_train_df = normalized_train[normalized_train.columns[normalized_train.columns!=predictor]]
    encoding_weight, z_weight= autoencoder(new_train_df, encoding_node, max_iter, learning_rate)
    z_weight_df = pd.DataFrame(z_weight)
    hidden_weight , output_weight = autoencoder_regression(z_weight_df, class_df,  hidden_node, max_iter, learning_rate)
    new_test_df = normalized_test.drop(predictor, axis=1)
    weight, hidden_layer1, hidden_layer2 = BP_encoding_regression(new_train_df, class_df, encoding_weight, hidden_weight, output_weight, max_iter, learning_rate) 
    backprop_prediction = predict_FFBP_regression(new_test_df, weight, encoding_node, hidden_node,  hidden_layer1, hidden_layer2)
    score= evaluation_metrics(test_df[predictor].values, backprop_prediction, method)
    score = round(score, 2)
    return score

# Autoencoder Feedforward backpropagation for 5-fold cross validation
# It uses 80% of the dataset for 5-fold cross validation, and gets the accuracy score for each fold.
def autoencoder_regression_fold(df, predictor, encoding_node, hidden_node,  max_iter, learning_rate, method):
    score_list = []
    for i in range(0, len(df)):
        k_validate = pd.concat([x for j,x in enumerate(df) if j!=i])
        k_validate = k_validate.reset_index(drop=True)
        score = autoencoding_FFBP_regression(k_validate, df[i], predictor, encoding_node, hidden_node, max_iter, learning_rate, method)
        score_list.append(score)
    return score_list

# Tuning Autoencoder Feedforward backpropagation for regression
# It uses 20% of the dataset for tuning. 
# It finds the best parameter combination of maximum iteration, learning rate, encoding node, hidden node.
def tuning_autoencoder_regression(df, max_iter_list, learning_rate_list, encoding_node_list, hidden_node_list, predictor, method):
    best_candidate = []
    parameter_list =[]
    for i in range(len(max_iter_list)):
        for j in range(len(learning_rate_list)):
            for k in range(len(encoding_node_list)):
                for m in range(len(hidden_node_list)):
                    parameter_list.append([i,j, k, m])
                    best_candidate.append(np.mean(autoencoder_regression_fold(df, 
                                                                               predictor, 
                                                                               encoding_node_list[k], 
                                                                               hidden_node_list[m], 
                                                                               max_iter_list[i], 
                                                                               learning_rate_list[j], 
                                                                               method)))     
    best_parameter =best_candidate.index(min(best_candidate))
    best_max_iter = max_iter_list[parameter_list[best_parameter][0]]
    best_learning_rate = learning_rate_list[parameter_list[best_parameter][1]]
    best_encoding_node =encoding_node_list[parameter_list[best_parameter][2]]
    best_hidden_node = hidden_node_list[parameter_list[best_parameter][3]]
    return best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node


# ## Breast Cancer Wisconsin Dataset

print("----------- Breast Cancer -----------")
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
validated_train_breast_cancer_size
#tuning
tuning_breast_cancer = cross_validation(test_breast_cancer_dataset,5)
tuning_breast_cancer_size = []
for i in range(0, 5):
    tuning_breast_cancer_size.append(tuning_breast_cancer [i].shape[0])
tuning_breast_cancer_size
# ### Breast Cancer Logistic Regression
learning_rate_list = [0.01, 0.05, 0.1]
max_iter_list = [10, 20, 50]
# best_learning_rate, best_max_iter = tuning_LogR_single(tuning_breast_cancer, 
#                                                         max_iter_list, 
#                                                        learning_rate_list, 
#                                                        'class', 
#                                                        "classification score")
best_max_iter, best_learning_rate = 50, 0.1
print("Logistic Regesssion Binary Classes")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
#Logistic Regesssion Binary Classes
#Best Max Iteration 50
#best_learning_rate 0.1
LogR_score_binary = LogR_binary_fold(validated_train_breast_cancer, 'class', best_max_iter , best_learning_rate, "classification score")
LogR_avg_score_binary  = round(np.mean(LogR_score_binary),2)
print("LogR binary score: " + str(LogR_score_binary) + "/ LogR binary avg score: "+ str(LogR_avg_score_binary ))
#LogR binary score: [77.6786, 82.8829, 81.0811, 64.8649, 84.6847]/ LogR binary avg score: 78.2384

# ### Breast Cancer FFBP
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_hidden_node = tuning_FFBP_classification(tuning_breast_cancer, max_iter_list, 
#                                                                                  learning_rate_list, hidden_node_list, 
#                                                                                  'class', 'classification score')
best_max_iter, best_learning_rate, best_hidden_node = 20, 0.1, 6
print("Feedforward with Backpropagation Classification")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_hidden_node", best_hidden_node)
# Feedforward with Backpropagation Classification
# Best Max Iteration 20
# best_learning_rate 0.1
# best_hidden_node 6
FFBP_class_score = FFBP_classification_fold(validated_train_breast_cancer, 'class',best_hidden_node, best_hidden_node, 
                                      best_max_iter , best_learning_rate, "classification score")
FFBP_avg_class_score = round(np.mean(FFBP_class_score),2)
print("FFBP score: " + str(FFBP_class_score) + "/ FFBP avg score: "+ str(FFBP_avg_class_score))
#FFBP score: [83.04, 98.2, 94.59, 97.3, 97.3]/ FFBP avg score: 94.09

# ### Breast Cancer Autoencoder
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
encoding_node_list = [2,3]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = tuning_autoencoder_classification(tuning_breast_cancer, max_iter_list, 
#                                                                                                      learning_rate_list, encoding_node_list,
#                                                                                                      hidden_node_list, 'class', 'classification score')

best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = 0.1, 10, 3, 5
print("Autoencoder")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_encoding_node", best_encoding_node)
print("best_hidden_node", best_hidden_node)
# Autoencoder
# Best Max Iteration 10
# best_learning_rate 0.1
# best_encoding_node 3
# best_hidden_node 5
autoencoder_class_score = autoencoder_classification_fold(validated_train_breast_cancer, 'class', best_encoding_node, 
                                                    best_hidden_node, best_max_iter , best_learning_rate, "classification score")
autoencoder_class_avg_score  = round(np.mean(autoencoder_class_score),2)
print("Autoencoder score: " + str(autoencoder_class_score) + "/ Autoencoder avg score: "+ str(autoencoder_class_avg_score))
#Autoencoder score: [85.71, 98.2, 94.59, 96.4, 98.2]/ Autoencoder avg score: 94.62


# ## Car Evaluation
print("----------- Car -----------")
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
validated_train_car_size

tuning_car = cross_validation(test_car_dataset,5)
tuning_car_size = []
for i in range(0, 5):
    tuning_car_size.append(tuning_car[i].shape[0])
tuning_car_size

# ### Car Logistic Regression
learning_rate_list = [0.01, 0.05, 0.1]
max_iter_list = [10, 20, 50]
# best_learning_rate, best_max_iter = tuning_LogR_multi(tuning_car, 
#                                                       max_iter_list, 
#                                                       learning_rate_list, 
#                                                       'class', 
#                                                        "classification score")
best_learning_rate, best_max_iter = 10, 0.01
print("Logistic Regesssion Multi Classes")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
# Logistic Regesssion Multi Classes
# Best Max Iteration 10
# best_learning_rate 0.01

LogR_score_multi = LogR_multi_fold(validated_train_car, 'class', best_max_iter , best_learning_rate, "classification score")
LogR_avg_score_multi  = round(np.mean(LogR_score_multi),2)
print("LogR multi score: " + str(LogR_score_multi) + "/ LogR multi avg score: "+ str(LogR_avg_score_multi))
#LogR multi score: [90.25, 79.35, 66.67, 65.58, 60.14]/ LogR multi avg score: 72.398

# ### Car FFBP
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_hidden_node = tuning_FFBP_classification(tuning_car, max_iter_list, 
#                                                                                  learning_rate_list, hidden_node_list, 
#                                                                                  'class', 'classification score')
best_learning_rate, best_max_iter, best_hidden_node = 0.1, 10, 7
print("Feedforward with Backpropagation Classification")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_hidden_node", best_hidden_node)
# Feedforward with Backpropagation Classification
# Best Max Iteration 10
# best_learning_rate 0.1
# best_hidden_node 7
FFBP_class_score = FFBP_classification_fold(validated_train_car, 'class',best_hidden_node, best_hidden_node, 
                                      best_max_iter , best_learning_rate, "classification score")
FFBP_avg_class_score = round(np.mean(FFBP_class_score),2)
print("FFBP score: " + str(FFBP_class_score) + "/ FFBP avg score: "+ str(FFBP_avg_class_score))
#FFBP score: [86.28, 86.59, 83.7, 86.96, 66.67]/ FFBP avg score: 82.04

# ### Car Autoencoder
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
encoding_node_list = [2,3]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = tuning_autoencoder_classification(tuning_car, max_iter_list, 
#                                                                                                      learning_rate_list, encoding_node_list,
#                                                                                                      hidden_node_list, 'class', 'classification score')

best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node=0.001, 10, 2, 5
print("Autoencoder")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_encoding_node", best_encoding_node)
print("best_hidden_node", best_hidden_node)
# Autoencoder
# Best Max Iteration 10
# best_learning_rate 0.001
# best_encoding_node 2
# best_hidden_node 5
autoencoder_class_score = autoencoder_classification_fold(validated_train_car, 'class', best_encoding_node, 
                                                    best_hidden_node, best_max_iter , best_learning_rate, "classification score")
autoencoder_class_avg_score  = round(np.mean(autoencoder_class_score),2)
print("Autoencoder score: " + str(autoencoder_class_score) + "/ Autoencoder avg score: "+ str(autoencoder_class_avg_score))
#Autoencoder score: [94.22, 79.35, 66.67, 65.58, 60.14]/ Autoencoder avg score: 73.19


# ## Congressional Vote
print("----------- Congressional Vote -----------")
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
validated_train_vote_size
tuning_vote = cross_validation(test_vote_dataset ,5)
tuning_vote_size = []
for i in range(0, 5):
    tuning_vote_size.append(tuning_vote[i].shape[0])
tuning_vote_size

# ### Vote Logistic Regression
learning_rate_list = [0.01, 0.05, 0.1]
max_iter_list = [10, 20, 50]
# best_learning_rate, best_max_iter = tuning_LogR_single(tuning_vote, 
#                                                         max_iter_list, 
#                                                        learning_rate_list, 
#                                                        'class', 
#                                                        "classification score")
best_learning_rate, best_max_iter = 0.1, 20                                                      
print("Logistic Regesssion Binary Classes")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
# Logistic Regesssion Binary Classes
# Best Max Iteration 20
# best_learning_rate 0.1

LogR_score_binary = LogR_binary_fold(validated_train_vote, 'class', best_max_iter , best_learning_rate, "classification score")
LogR_avg_score_binary  = round(np.mean(LogR_score_binary),2)
print("LogR binary score: " + str(LogR_score_binary) + "/ LogR binary avg score: "+ str(LogR_avg_score_binary ))
#LogR binary score: [98.57, 92.86, 95.71, 94.29, 97.06]/ LogR binary avg score: 95.7

# ### Vote FFBP
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_hidden_node = tuning_FFBP_classification(tuning_vote, max_iter_list, 
#                                                                                  learning_rate_list, hidden_node_list, 
#                                                                                  'class', 'classification score')
best_learning_rate, best_max_iter, best_hidden_node = 0.1, 20, 6
print("Feedforward with Backpropagation Classification")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_hidden_node", best_hidden_node)
# Feedforward with Backpropagation Classification
# Best Max Iteration 20
# best_learning_rate 0.1
# best_hidden_node 6
FFBP_class_score = FFBP_classification_fold(validated_train_vote, 'class',best_hidden_node, best_hidden_node, 
                                      best_max_iter , best_learning_rate, "classification score")
FFBP_avg_class_score = round(np.mean(FFBP_class_score),2)
print("FFBP score: " + str(FFBP_class_score) + "/ FFBP avg score: "+ str(FFBP_avg_class_score))
#FFBP score: [98.57, 94.29, 95.71, 94.29, 98.53]/ FFBP avg score: 96.28

# ### Vote Autoencoder
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
encoding_node_list = [2,3]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = tuning_autoencoder_classification(tuning_vote, max_iter_list, 
#                                                                                                      learning_rate_list, encoding_node_list,
#                                                                                                      hidden_node_list, 'class', 'classification score')
best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = 0.1, 20, 2, 7
print("Autoencoder")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_encoding_node", best_encoding_node)
print("best_hidden_node", best_hidden_node)
# Autoencoder
# Best Max Iteration 20
# best_learning_rate 0.1
# best_encoding_node 2
# best_hidden_node 7
autoencoder_class_score = autoencoder_classification_fold(validated_train_vote, 'class', best_encoding_node, 
                                                    best_hidden_node, best_max_iter , best_learning_rate, "classification score")
autoencoder_class_avg_score  = round(np.mean(autoencoder_class_score),2)
print("Autoencoder score: " + str(autoencoder_class_score) + "/ Autoencoder avg score: "+ str(autoencoder_class_avg_score))
#Autoencoder score: [97.14, 94.29, 94.29, 94.29, 98.53]/ Autoencoder avg score: 95.71


# ## Abalone

print("----------- Abalone -----------")
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
validated_train_abalone_size

tuning_abalone = cross_validation_regression(test_abalone_dataset, 5, 'rings')
tuning_abalone_size = []
for i in range(0, 5):
    tuning_abalone_size.append(tuning_abalone[i].shape[0])
tuning_abalone_size

# ### Abalone Linear Regression
learning_rate_list = [0.01, 0.05, 0.1, 0.5]

best_learning_rate = tuning_LinR(tuning_abalone, learning_rate_list, 'rings', "MSE")
print("Linear Regression")
print("best_learning_rate", best_learning_rate)
#best_learning_rate 0.05

LinR_score = LinR_fold(validated_train_abalone, 'rings', best_learning_rate, 'MSE')
LinR_avg_score = round(np.mean(LinR_score),2)
print("LinR binary score: " + str(LinR_score) + "/ LinR binary avg score: "+ str(LinR_avg_score))
#LinR binary score: [19.0, 4.77, 10.11, 15.75, 100.10]/ LinR binary avg score: 29.94

# ### Abalone FFBP
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_hidden_node = tuning_FFBP_regression(tuning_abalone, max_iter_list, 
#                                                                                  learning_rate_list, hidden_node_list, 
#                                                                                  'rings', 'MSE')
best_learning_rate, best_max_iter, best_hidden_node = 0.001, 10, 5
print("Feedforward with Backpropagation Regression")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_hidden_node", best_hidden_node)
# Feedforward with Backpropagation Regression
# Best Max Iteration 10
# best_learning_rate 0.001
# best_hidden_node 5
FFBP_reg_score = FFBP_regression_fold(validated_train_abalone, 'rings',best_hidden_node, best_hidden_node, 
                                      best_max_iter , best_learning_rate, "MSE")
FFBP_reg_avg_score  = round(np.mean(FFBP_reg_score),2)
print("FFBP score: " + str(FFBP_reg_score) + "/ FFBP avg score: "+ str(FFBP_reg_avg_score))
#FFBP score: [128.02, 82.15, 62.02, 40.08, 21.48]/ FFBP avg score: 66.75

# ### Abalone Autoencoder
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
encoding_node_list = [2,3]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = tuning_autoencoder_regression(tuning_abalone, max_iter_list, 
#                                                                                                      learning_rate_list, encoding_node_list,
#                                                                                                      hidden_node_list, 'rings', 'MSE')

best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = 0.001, 10, 2, 5
print("Autoencoder")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_encoding_node", best_encoding_node)
print("best_hidden_node", best_hidden_node)
# Best learning rate 0.001
# best max iteration 20
# best encoding node 2
# best hidden node 5
autoencoder_reg_score = autoencoder_regression_fold(validated_train_abalone, 'rings', best_encoding_node, 
                                                    best_hidden_node, best_max_iter , best_learning_rate, "MSE")
autoencoder_reg_avg_score  = round(np.mean(autoencoder_reg_score),2)
print("Autoencoder score: " + str(autoencoder_reg_score) + "/ Autoencoder avg score: "+ str(autoencoder_reg_avg_score))
#Autoencoder score: [127.95, 82.15, 62.02, 40.08, 21.48]/ Autoencoder avg score: 66.74

# ## Computer Hardware
print("----------- Computer Hardware -----------")
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
validated_train_computer_size
tuning_computer = cross_validation_regression(test_computer_dataset , 5, 'PRP')
tuning_computer_size = []
for i in range(0, 5):
    tuning_computer_size.append(tuning_computer[i].shape[0])
tuning_computer_size

# ### Hardware Linear Regression
learning_rate_list = [0.01, 0.05, 0.1, 0.5]

best_learning_rate = tuning_LinR(tuning_computer, learning_rate_list, 'PRP', "MSE")
print("Linear Regression")
print("best_learning_rate", best_learning_rate)
#best_learning_rate 0.5
LinR_score = LinR_fold(validated_train_computer, 'PRP', best_learning_rate, 'MSE')
LinR_avg_score = round(np.mean(LinR_score),4)
print("LinR binary score: " + str(LinR_score) + "/ LinR binary avg score: "+ str(LinR_avg_score))
#LinR binary score: [0.12, 0.47, 0.49, 0.87, 10.96]/ LinR binary avg score: 2.58

# ### Hardware FFBP
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_hidden_node = tuning_FFBP_regression(tuning_computer, max_iter_list, 
#                                                                                  learning_rate_list, hidden_node_list, 
#                                                                                  'PRP', 'MSE')
best_learning_rate, best_max_iter, best_hidden_node = 0.001, 20, 6
print("Feedforward with Backpropagation Regression")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_hidden_node", best_hidden_node)
# Feedforward with Backpropagation Regression
# Best Max Iteration 20
# best_learning_rate 0.001
# best_hidden_node 6
FFBP_reg_score = FFBP_regression_fold(validated_train_computer, 'PRP',best_hidden_node, best_hidden_node, 
                                      best_max_iter , best_learning_rate, "MSE")
FFBP_reg_avg_score  = round(np.mean(FFBP_reg_score),2)
print("FFBP score: " + str(FFBP_reg_score) + "/ FFBP avg score: "+ str(FFBP_reg_avg_score))
#FFBP score: [2.78, 0.47, 0.47, 0.3, 8.6]/ FFBP avg score: 2.52

# ### Hardware Autoencoder
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
encoding_node_list = [2,3]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = tuning_autoencoder_regression(tuning_computer, max_iter_list, 
#                                                                                                      learning_rate_list, encoding_node_list,
#                                                                                                      hidden_node_list, 'PRP', 'MSE')
best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = 0.001, 20, 2, 5
print("Autoencoder")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_encoding_node", best_encoding_node)
print("best_hidden_node", best_hidden_node)
# Autoencoder
# Best Max Iteration 20
# best_learning_rate 0.001
# best_encoding_node 2
# best_hidden_node 5
autoencoder_reg_score = autoencoder_regression_fold(validated_train_computer, 'PRP', best_encoding_node, 
                                                    best_hidden_node, best_max_iter , best_learning_rate, "MSE")
autoencoder_reg_avg_score  = round(np.mean(autoencoder_reg_score),2)
print("Autoencoder score: " + str(autoencoder_reg_score) + "/ Autoencoder avg score: "+ str(autoencoder_reg_avg_score))
#Autoencoder score: [2.72, 0.43, 0.43, 0.28, 8.56]/ Autoencoder avg score: 2.48


# ## Forest Fires
print("----------- Forest Fires -----------")
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
validated_train_forest_size 

tuning_forest = cross_validation_regression(test_forest_dataset, 5, 'area')
tuning_forest_size  = []
for i in range(0, 5):
    tuning_forest_size .append(tuning_forest[i].shape[0])
tuning_forest_size 

learning_rate_list = [0.01, 0.05, 0.1, 0.5]
# best_learning_rate = tuning_LinR(tuning_forest, learning_rate_list, 'area', "MSE")
best_learning_rate = 0.05
print("Linear Regression")
print("best_learning_rate", best_learning_rate)
#best_learning_rate 0.05
LinR_score = LinR_fold(validated_train_forest, 'area', best_learning_rate, 'MSE')
LinR_avg_score = round(np.mean(LinR_score),2)
print("LinR binary score: " + str(LinR_score) + "/ LinR binary avg score: "+ str(LinR_avg_score))
#LinR binary score: [0.19, 0.26, 0.14, 1.74, 10.5]/ LinR binary avg score: 2.56

# ### Forest Fires FFBP
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_hidden_node = tuning_FFBP_regression(tuning_forest, max_iter_list, 
#                                                                                  learning_rate_list, hidden_node_list, 
#                                                                                  'area', 'MSE')
best_learning_rate, best_max_iter, best_hidden_node = 0.001, 10, 5
print("Feedforward with Backpropagation Regression")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_hidden_node", best_hidden_node)
# Feedforward with Backpropagation Regression
# Best Max Iteration 10
# best_learning_rate 0.001
# best_hidden_node 5
FFBP_reg_score = FFBP_regression_fold(validated_train_forest, 'area',best_hidden_node, best_hidden_node, 
                                      best_max_iter , best_learning_rate, "MSE")
FFBP_reg_avg_score  = round(np.mean(FFBP_reg_score),2)
print("FFBP score: " + str(FFBP_reg_score) + "/ FFBP avg score: "+ str(FFBP_reg_avg_score))
#FFBP score: [3.05, 3.06, 1.88, 0.32, 7.48]/ FFBP avg score: 3.16

# ### Forest Fires Autoencoder
max_iter_list = [10, 20]
learning_rate_list = [0.001, 0.005, 0.01, 0.05, 0.1]
encoding_node_list = [2,3]
hidden_node_list = [5, 6, 7]
# best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = tuning_autoencoder_regression(tuning_forest, max_iter_list, 
#                                                                                                      learning_rate_list, encoding_node_list,
#                                                                                                      hidden_node_list, 'area', 'MSE')
best_learning_rate, best_max_iter, best_encoding_node, best_hidden_node = 0.001, 10, 2, 5
print("Autoencoder")
print('Best Max Iteration', best_max_iter)
print("best_learning_rate", best_learning_rate)
print("best_encoding_node", best_encoding_node)
print("best_hidden_node", best_hidden_node)
# Autoencoder
# Best Max Iteration 10
# best_learning_rate 0.001
# best_encoding_node 2
# best_hidden_node 5
autoencoder_reg_score = autoencoder_regression_fold(validated_train_forest, 'area', best_encoding_node, 
                                                    best_hidden_node, best_max_iter , best_learning_rate, "MSE")
autoencoder_reg_avg_score  = round(np.mean(autoencoder_reg_score),2)
print("Autoencoder score: " + str(autoencoder_reg_score) + "/ Autoencoder avg score: "+ str(autoencoder_reg_avg_score))
#Autoencoder score: [2.98, 3.03, 1.83, 0.33, 7.48]/ Autoencoder avg score: 3.13

