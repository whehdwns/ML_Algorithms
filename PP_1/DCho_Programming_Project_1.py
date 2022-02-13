import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import os
import glob

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
# 1. Split dataset into training and testset using split_dataset function.
# 2. With training set, it splits training set into k-equaled size. 
# 3. It returns k-equal-sized training partitions, and test set. 

def cross_validation(df, percent, k):
    train, test  = split_dataset(df, percent)
    train_size= len(train)
    fold_size = train_size//k 
    remainder = train_size %k
    train_folds = []
    start = 0
    for i in range(0,k):
        if i < remainder:
            fold =  train.iloc[start : start+fold_size+1]
            train_folds.append(fold)
            start +=  fold_size + 1
        else:
            fold =  train.iloc[start : start+fold_size]
            train_folds.append(fold)
            start +=  fold_size
    return train_folds, test


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
        if len(y_pred) ==len(y_true):
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

# For this project, it demonstrate the evaluation metrics for classification and regression 
# by averaging the predictions over all of the folds of the dataset.

def evaluation_metrics_for_classification_and_regression(prediction):
    return round(np.mean(prediction),2)


# ## Naive Majority Predictor Algorithm for Classification

# Naive Majority predictor Algorithm
# 1. Using training valiation set from 5-fold cross-valiation, 
#    it rotates the fold to use training set for 4 fold, and remaining fold for test set. 
# 2. It finds the most common label from each rotating 4 fold.
# 3. It counts the most common labels from previous steps. 
# 4. It divides the counts that matches with most common labels to calcualte the accuracy.


def majority_predictor_classification(train_val, predictor):
    accuracy = []
    print("Accuracy for Each Fold")
    for i in range(0, len(train_val)):
        k_validate = pd.concat([x for j,x in enumerate(train_val) if j!=i])
        count = 0
        for k in range(len(train_val[i])):
            if train_val[i][predictor].values[k] == k_validate[predictor].value_counts().index[0]:
                count +=1
        result = count / len(train_val[i])
        result = round(result*100, 2)
        accuracy.append(result)
        print("Fold " + str(i+1) + " : " + str(result) +"%")
    return accuracy


# ## Naive Majority Predictor Algorithm for Regression

# Naive Mean Regressor Algorithm
# 1. Using training valiation set from 5-fold cross-valiation, 
#    it rotates the fold to use training set for 4 fold, and remaining fold for test set. 
# 2. It finds the mean values from each rotating 4 fold.
# 3. It counts the mean values from previous steps. 
# 4. It divides the counts that matches with mean values to calcualte the accuracy.

def majority_predictor_regression(train_val, predictor):
    accuracy = []
    for i in range(0, len(train_val)):
        k_validate = pd.concat([x for j,x in enumerate(train_val) if j!=i])
        count = 0
        for k in range(len(train_val[i])):
            if train_val[i][predictor].values[k] == round(k_validate[predictor].mean()):
                count +=1
        result = count / len(train_val[i])
        result = round(result*100, 2)
        accuracy.append(result)
        print("Fold " + str(i+1) + " : " + str(result) +"%")
    return accuracy


# ## Breast Cancer Wisconsin Dataset
print("Breast Cancer Dataset")
#load Data
breast_cancer_dataset = open_csv_dataset('breast-cancer', True)
breast_cancer_dataset.head(3)
# Handling Missing Data
clean_breast_cancer_dataset = handling_missing_values(breast_cancer_dataset)
clean_breast_cancer_dataset.isnull().sum().any()
clean_breast_cancer_dataset.head()
#Drop unnecessary columns for data
breast_cancer_dataset_v1  = clean_breast_cancer_dataset.copy()
breast_cancer_dataset_v1 = breast_cancer_dataset_v1.drop(['sample_code_number'], axis=1)
breast_cancer_dataset_v1.head(3)
breast_cancer_dataset_v1.describe()
#Split Dataset
train_breast_cancer_dataset, test_breast_cancer_dataset = split_dataset(breast_cancer_dataset_v1, 0.8)
#Standardization
train_breast_cancer_zscore_dataset, test_breast_cancer_zscore_dataset  = Standardization(train_breast_cancer_dataset,
                                                                                         test_breast_cancer_dataset)
train_breast_cancer_zscore_dataset.head(3)
test_breast_cancer_zscore_dataset.head(3)

#5-kold cross validation
validated_train_breast_cancer, validated_test_breast_cancer = cross_validation(breast_cancer_dataset_v1,0.8,5)
validated_test_breast_cancer.shape[0]
validated_train_breast_cancer_size = []
#checking size of 5-fold
for i in range(0, 5):
    validated_train_breast_cancer_size.append(validated_train_breast_cancer[i].shape[0])
validated_train_breast_cancer_size
#Majority Predictor
breast_cancer_accuracy = majority_predictor_classification(validated_train_breast_cancer, 'class')
print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(breast_cancer_accuracy))+ "%")



# ## Car Evaluation

print("Car Evaluation")
#Load data
car_dataset = open_csv_dataset('car', True)
car_dataset.head(3)
#Handling Missing Values
clean_car_dataset = handling_missing_values(car_dataset)
clean_car_dataset.isnull().sum().any()
clean_car_dataset.head(3)
#Handdling Categorical values
categorized_car_datast = replacing_string_to_numeric_all_columns(clean_car_dataset)
car_dataset_v1  = categorized_car_datast.copy()
car_dataset_v1.describe()
#Split Dataset 
train_car_dataset, test_car_dataset = split_dataset(car_dataset_v1, 0.8)
#Standardization
train_car_zscore_dataset,test_car_zscore_dataset  = Standardization(train_car_dataset,test_car_dataset)
train_car_zscore_dataset.head()
test_car_zscore_dataset.head()
#5-fold cross validation
validated_train_car, validated_test_car = cross_validation(car_dataset_v1,0.8,5)
validated_test_car.shape[0]
#size of 5-fold
validated_train_car_size = []
for i in range(0, 5):
    validated_train_car_size.append(validated_train_car[i].shape[0])
validated_train_car_size
#Majority predictor
car_accuracy = majority_predictor_classification(validated_train_car, 'class')
print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(car_accuracy))+ "%")


# ## Congressional Vote
print("Congressional Vote")
#load Data
vote_dataset = open_csv_dataset('vote', True)
vote_dataset.head()
vote_dataset.isnull().sum().any()
#Categorical Data
categorized_vote_datast = replacing_string_to_numeric_all_columns(vote_dataset)
categorized_vote_datast.head()
vote_dataset_v1  = categorized_vote_datast.copy()
vote_dataset_v1.describe()
#Split Dataset
train_vote_dataset, test_vote_dataset = split_dataset(vote_dataset_v1, 0.8)
#Standardization
train_vote_zscore_dataset, test_vote_zscore_dataset  = Standardization(train_vote_dataset,test_vote_dataset)
train_vote_zscore_dataset.head()
test_vote_zscore_dataset.head()
#5-fold cross validation
validated_train_vote, validated_test_vote = cross_validation(vote_dataset_v1,0.8,5)
validated_test_vote.shape[0]
#Size of 5-fold
validated_train_vote_size = []
for i in range(0, 5):
    validated_train_vote_size.append(validated_train_vote[i].shape[0])
validated_train_vote_size
#Majority predictor
vote_accuracy = majority_predictor_classification(validated_train_vote, 'class')
print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(vote_accuracy))+ "%")


# ## Abalone

print("Abalone")
#Load Data
abalone_dataset = open_csv_dataset('abalone', True)
abalone_dataset.head(3)
abalone_dataset.isnull().sum().any()
#Categorical Data
categorized_abalone_datast = replacing_string_to_numeric_multiple_columns(abalone_dataset, ['sex'])
abalone_dataset_v1  = categorized_abalone_datast.copy()
abalone_dataset_v1.head()
abalone_discretize_dataset = abalone_dataset_v1.copy()
#Discretization
abalone_discretize_dataset = discretization(abalone_discretize_dataset, 5, 'equal_frequency',
                                           ['length', 'diameter', 'height', 'whole_height',
                                            'shucked_height', 'viscera_weight', 'shell_weight', 'rings'])
abalone_dataset_v2 =  abalone_discretize_dataset.copy()
abalone_dataset_v2.describe()
#Split Dataset
train_abalone_dataset, test_abalone_dataset = split_dataset(abalone_dataset_v2, 0.8)
#Standardization
train_abalone_zscore_dataset,test_abalone_zscore_dataset  = Standardization(train_abalone_dataset,
                                                                             test_abalone_dataset)
train_abalone_zscore_dataset.head()
test_abalone_zscore_dataset.head()
#5-fold cross validation
validated_train_abalone, validated_test_abalone = cross_validation(abalone_dataset_v2,0.8,5)
validated_test_abalone.shape[0]
#Size of 5-fold
validated_train_abalone_size = []
for i in range(0, 5):
    validated_train_abalone_size.append(validated_train_abalone[i].shape[0])
validated_train_abalone_size
#Majority Predictor for regression
abalone_accuracy = majority_predictor_regression(validated_train_abalone, 'rings')
print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(abalone_accuracy))+ "%")


# ## Computer Hardware

print("Computer Hardware")
#Load Data
computer_dataset = open_csv_dataset('machine', True)
computer_dataset.head()
computer_dataset.isnull().sum().any()
#Drop unnecessary column
computer_ERP  = computer_dataset['ERP']
computer_dataset = computer_dataset.drop(['vendor', 'model', 'ERP'], axis = 1)
computer_dataset_v1  = computer_dataset.copy()
computer_dataset_v1.head()
computer_discretize_dataset = computer_dataset_v1.copy()
#Discretization
computer_discretize_dataset = discretization(computer_discretize_dataset, 
                                             5, 
                                             'equal_frequency', 
                                             ['myct','mmin', 'mmax', 'cach', 'chmin', 'chmax', 'PRP'])
computer_discretize_dataset.head()
computer_dataset_v2 =  computer_discretize_dataset.copy()
computer_dataset_v2.describe()
#Split dataset
train_computer_dataset, test_computer_dataset = split_dataset(computer_dataset_v2, 0.8)
#Standardization
train_computer_zscore_dataset, test_computer_zscore_dataset  = Standardization(train_computer_dataset,
                                                                               test_computer_dataset)
train_computer_zscore_dataset.head()
test_computer_zscore_dataset.head()
#5-fold cross validation
validated_train_computer, validated_test_computer = cross_validation(computer_dataset_v2,0.8,5)
validated_test_computer.shape[0]
#size of 5-fold
validated_train_computer_size = []
for i in range(0, 5):
    validated_train_computer_size.append(validated_train_computer[i].shape[0])
validated_train_computer_size
#Majority Predictor for Regression
computer_accuracy =majority_predictor_regression(validated_train_computer, 'PRP')
print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(computer_accuracy))+ "%")


# ## Forest Fires
print("Forest Fires")
#Load Data
forest_dataset = open_csv_dataset('forest', False)
forest_dataset.head()
forest_dataset.isnull().sum().any()
#Categorical values
categorized_forest_dataset = replacing_string_to_numeric_multiple_columns(forest_dataset,['month', 'day'])
categorized_forest_dataset.head()
forest_dataset_v1  = categorized_forest_dataset.copy()
#Log Transform
forest_dataset_v1['area'] = log_transform(forest_dataset_v1['area'])
forest_dataset_v1.head()
forest_discretize_dataset = forest_dataset_v1.copy()
#Discretization
forest_discretize_dataset = discretization(forest_discretize_dataset, 
                                           5, 
                                           'equal_frequency', 
                                           ['FFMC', 'DMC', 'DC', 'ISI','temp','RH', 'wind', 'rain', 'area'])
forest_dataset_v2 =  forest_discretize_dataset.copy()
forest_dataset_v2.describe()
#SPlit Dataset
train_forest_dataset, test_forest_dataset = split_dataset(forest_dataset_v2, 0.8)
#Standardization
train_forest_zscore_dataset, test_forest_zscore_dataset  = Standardization(train_forest_dataset,test_forest_dataset)
train_forest_zscore_dataset.head()
test_forest_zscore_dataset.head()
#5-fold cross validation
validated_train_forest, validated_test_forest = cross_validation(forest_dataset_v2,0.8,5)
validated_test_forest.shape[0]
#Size of 5-fold
validated_train_forest_size = []
for i in range(0, 5):
    validated_train_forest_size.append(validated_train_forest[i].shape[0])
validated_train_forest_size
#Majority Predictor for regression
forest_accuracy = majority_predictor_regression(validated_train_forest, 'area')
print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(forest_accuracy))+ "%")

