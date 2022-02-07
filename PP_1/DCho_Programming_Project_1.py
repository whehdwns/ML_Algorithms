#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import os
import glob


# In[2]:


path = os.getcwd()
all_files = glob.glob(path + "/dataset/*.data")
filesnames = os.listdir('dataset/')


# In[3]:


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

# In[4]:


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

# In[5]:


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

# In[6]:


# Unique value in columns
# 1. It categorizes all columns in the dataset. 

def view_unique_value_in_columns(df):
    for i in df.columns.values:
        print(categorize_dataset(df, i))


# In[7]:


# Unique value in single columns
# 1. It categorizes single columns in the dataset. 

def view_unique_value_in_single_column(df, columns):
    print(categorize_dataset(df, columns))


# In[8]:


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


# In[9]:


# Replace to numeric single/multiple columns in the dataset
# 1. Based on categorize dataset function, it replaces string value to numeric values for single/multiple columns

def replacing_string_to_numeric_multiple_columns(df, string_columns ):
    for i in string_columns:
        df = df.replace(categorize_dataset(df,i))
    return df


# In[10]:


# Replace to numeric all columns in the dataset
# 1. Based on categorize dataset function, it replaces string value to numeric values for all columns.

def replacing_string_to_numeric_all_columns(df):
    for i in df.columns.values:
        df = df.replace(categorize_dataset(df, i))
    return df


# ## Log Transform

# In[11]:


# Log Transform
# 1. This function is used to apply for Forest Fires data. 
# 2. Based on note, it shows the output area is very skewed toward 0.0. The authors recommend a log transform.
# 3. It log transform certain columns in the dataset. 

def log_transform(x):
    return np.log(x + 1)


# ## Discretization

# In[12]:


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

# In[13]:


# Split dataset
# 1. It splits the dataset into two: training set and test set.
# 2. Training set has 80% of original dataset. 
# 3. Testing set has 20% of original dataset. 

def split_dataset(df, train_perc):
    train_end_ind = int(round(df.shape[0] * train_perc))
    train = df.iloc[0:train_end_ind]
    test = df.iloc[train_end_ind:-1]
    return (train, test)


# In[14]:


# Z-score standardization
# It computes z-score by (observed value - mean of the sample) / standard deviation of the sample

def z_score_standardization(df):
    z_score = (df-df.mean())/df.std()
    return z_score


# In[15]:


# Standardization
# 1. It applies z-score standardization for training set and testign set. 

def Standardization(training, testing):
    training_zscore = z_score_standardization(training)
    testing_zscore = z_score_standardization(testing)
    return (training_zscore, testing_zscore)


# ## Cross-validation

# In[16]:


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

# In[17]:


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


# In[18]:


# For this project, it demonstrate the evaluation metrics for classification and regression 
# by averaging the predictions over all of the folds of the dataset.

def evaluation_metrics_for_classification_and_regression(prediction):
    return round(np.mean(prediction),2)


# ## Naive Majority Predictor Algorithm for Classification

# In[19]:


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

# In[20]:


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

# In[21]:

print("Breast Cancer Dataset")
breast_cancer_dataset = open_csv_dataset('breast-cancer', True)
breast_cancer_dataset.head(3)


# In[22]:


clean_breast_cancer_dataset = handling_missing_values(breast_cancer_dataset)


# In[23]:


clean_breast_cancer_dataset.isnull().sum().any()


# In[24]:


clean_breast_cancer_dataset.head()


# In[25]:


breast_cancer_dataset_v1  = clean_breast_cancer_dataset.copy()
breast_cancer_dataset_v1 = breast_cancer_dataset_v1.drop(['sample_code_number'], axis=1)


# In[26]:


breast_cancer_dataset_v1.head(3)


# In[27]:


#breast_cancer_dataset_v1.boxplot(figsize=(20,3))


# In[28]:


breast_cancer_dataset_v1.describe()


# In[29]:


train_breast_cancer_dataset, test_breast_cancer_dataset = split_dataset(breast_cancer_dataset_v1, 0.8)


# In[30]:


train_breast_cancer_zscore_dataset, test_breast_cancer_zscore_dataset  = Standardization(train_breast_cancer_dataset,
                                                                                         test_breast_cancer_dataset)


# In[31]:


train_breast_cancer_zscore_dataset.head(3)


# In[32]:


test_breast_cancer_zscore_dataset.head(3)


# In[33]:


validated_train_breast_cancer, validated_test_breast_cancer = cross_validation(breast_cancer_dataset_v1,0.8,5)


# In[34]:


validated_test_breast_cancer.shape[0]


# In[35]:


validated_train_breast_cancer_size = []
for i in range(0, 5):
    validated_train_breast_cancer_size.append(validated_train_breast_cancer[i].shape[0])
validated_train_breast_cancer_size


# In[36]:


breast_cancer_accuracy = majority_predictor_classification(validated_train_breast_cancer, 'class')


# In[37]:


print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(breast_cancer_accuracy))+ "%")


# ## Car Evaluation

# In[38]:

print("Car Evaluation")
car_dataset = open_csv_dataset('car', True)
car_dataset.head(3)


# In[39]:


clean_car_dataset = handling_missing_values(car_dataset)


# In[40]:


clean_car_dataset.isnull().sum().any()


# In[41]:


clean_car_dataset.head(3)


# In[42]:


# view_unique_value_in_columns(clean_car_dataset)


# In[43]:


categorized_car_datast = replacing_string_to_numeric_all_columns(clean_car_dataset)


# In[44]:


car_dataset_v1  = categorized_car_datast.copy()
car_dataset_v1.head()


# In[45]:


# car_dataset_v1.boxplot(figsize=(20,3))


# In[46]:


car_dataset_v1.describe()


# In[47]:


train_car_dataset, test_car_dataset = split_dataset(car_dataset_v1, 0.8)


# In[48]:


train_car_zscore_dataset,test_car_zscore_dataset  = Standardization(train_car_dataset,test_car_dataset)


# In[49]:


train_car_zscore_dataset.head()


# In[50]:


test_car_zscore_dataset.head()


# In[51]:


validated_train_car, validated_test_car = cross_validation(car_dataset_v1,0.8,5)


# In[52]:


validated_test_car.shape[0]


# In[53]:


validated_train_car_size = []
for i in range(0, 5):
    validated_train_car_size.append(validated_train_car[i].shape[0])
validated_train_car_size


# In[54]:


car_accuracy = majority_predictor_classification(validated_train_car, 'class')


# In[55]:


print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(car_accuracy))+ "%")


# ## Congressional Vote

# In[56]:

print("Congressional Vote")
vote_dataset = open_csv_dataset('vote', True)
vote_dataset.head()


# In[57]:


vote_dataset.isnull().sum().any()


# In[58]:


categorized_vote_datast = replacing_string_to_numeric_all_columns(vote_dataset)
categorized_vote_datast.head()


# In[59]:


vote_dataset_v1  = categorized_vote_datast.copy()


# In[60]:


# vote_dataset_v1.boxplot(figsize=(20,3))


# In[61]:


vote_dataset_v1.describe()


# In[62]:


train_vote_dataset, test_vote_dataset = split_dataset(vote_dataset_v1, 0.8)


# In[63]:


train_vote_zscore_dataset, test_vote_zscore_dataset  = Standardization(train_vote_dataset,test_vote_dataset)


# In[64]:


train_vote_zscore_dataset.head()


# In[65]:


test_vote_zscore_dataset.head()


# In[66]:


validated_train_vote, validated_test_vote = cross_validation(vote_dataset_v1,0.8,5)


# In[67]:


validated_test_vote.shape[0]


# In[68]:


validated_train_vote_size = []
for i in range(0, 5):
    validated_train_vote_size.append(validated_train_vote[i].shape[0])
validated_train_vote_size


# In[69]:


vote_accuracy = majority_predictor_classification(validated_train_vote, 'class')


# In[70]:


print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(vote_accuracy))+ "%")


# ## Abalone

# In[71]:

print("Abalone")
abalone_dataset = open_csv_dataset('abalone', True)
abalone_dataset.head(3)


# In[72]:


abalone_dataset.isnull().sum().any()


# In[73]:


categorized_abalone_datast = replacing_string_to_numeric_multiple_columns(abalone_dataset, ['sex'])


# In[74]:


abalone_dataset_v1  = categorized_abalone_datast.copy()


# In[75]:


abalone_dataset_v1.head()


# In[76]:


abalone_discretize_dataset = abalone_dataset_v1.copy()


# In[77]:


abalone_discretize_dataset = discretization(abalone_discretize_dataset, 5, 'equal_frequency',
                                           ['length', 'diameter', 'height', 'whole_height',
                                            'shucked_height', 'viscera_weight', 'shell_weight', 'rings'])


# In[78]:


abalone_dataset_v2 =  abalone_discretize_dataset.copy()
abalone_dataset_v2.tail(5)


# In[79]:


# abalone_dataset_v2.boxplot(figsize=(20,3))


# In[80]:


abalone_dataset_v2.describe()


# In[81]:


train_abalone_dataset, test_abalone_dataset = split_dataset(abalone_dataset_v2, 0.8)


# In[82]:


train_abalone_zscore_dataset,test_abalone_zscore_dataset  = Standardization(train_abalone_dataset,
                                                                             test_abalone_dataset)


# In[83]:


train_abalone_zscore_dataset.head()


# In[84]:


test_abalone_zscore_dataset.head()


# In[85]:


validated_train_abalone, validated_test_abalone = cross_validation(abalone_dataset_v2,0.8,5)


# In[86]:


validated_test_abalone.shape[0]


# In[87]:


validated_train_abalone_size = []
for i in range(0, 5):
    validated_train_abalone_size.append(validated_train_abalone[i].shape[0])
validated_train_abalone_size


# In[88]:


abalone_accuracy = majority_predictor_regression(validated_train_abalone, 'rings')


# In[89]:


print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(abalone_accuracy))+ "%")


# ## Computer Hardware

# In[90]:

print("Computer Hardware")
computer_dataset = open_csv_dataset('machine', True)
computer_dataset.head()


# In[91]:


computer_dataset.isnull().sum().any()


# In[92]:


computer_ERP  = computer_dataset['ERP']
computer_dataset = computer_dataset.drop(['vendor', 'model', 'ERP'], axis = 1)


# In[93]:


computer_dataset_v1  = computer_dataset.copy()
computer_dataset_v1.head()


# In[94]:


computer_discretize_dataset = computer_dataset_v1.copy()


# In[95]:


computer_discretize_dataset = discretization(computer_discretize_dataset, 
                                             5, 
                                             'equal_frequency', 
                                             ['myct','mmin', 'mmax', 'cach', 'chmin', 'chmax', 'PRP'])


# In[96]:


computer_discretize_dataset.head()


# In[97]:


computer_dataset_v2 =  computer_discretize_dataset.copy()


# In[98]:


computer_dataset_v2.describe()


# In[99]:


train_computer_dataset, test_computer_dataset = split_dataset(computer_dataset_v2, 0.8)


# In[100]:


train_computer_zscore_dataset, test_computer_zscore_dataset  = Standardization(train_computer_dataset,
                                                                               test_computer_dataset)


# In[101]:


train_computer_zscore_dataset.head()


# In[102]:


test_computer_zscore_dataset.head()


# In[103]:


validated_train_computer, validated_test_computer = cross_validation(computer_dataset_v2,0.8,5)


# In[104]:


validated_test_computer.shape[0]


# In[105]:


validated_train_computer_size = []
for i in range(0, 5):
    validated_train_computer_size.append(validated_train_computer[i].shape[0])
validated_train_computer_size


# In[106]:


computer_accuracy =majority_predictor_regression(validated_train_computer, 'PRP')


# In[107]:


print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(computer_accuracy))+ "%")


# ## Forest Fires

# In[108]:

print("Forest Fires")
forest_dataset = open_csv_dataset('forest', False)
forest_dataset.head()


# In[109]:


forest_dataset.isnull().sum().any()


# In[110]:


categorized_forest_dataset = replacing_string_to_numeric_multiple_columns(forest_dataset,['month', 'day'])
categorized_forest_dataset.head()


# In[111]:


forest_dataset_v1  = categorized_forest_dataset.copy()
forest_dataset_v1['area'] = log_transform(forest_dataset_v1['area'])
forest_dataset_v1.head()


# In[112]:


forest_discretize_dataset = forest_dataset_v1.copy()


# In[113]:


forest_discretize_dataset = discretization(forest_discretize_dataset, 
                                           5, 
                                           'equal_frequency', 
                                           ['FFMC', 'DMC', 'DC', 'ISI','temp','RH', 'wind', 'rain', 'area'])


# In[114]:


forest_dataset_v2 =  forest_discretize_dataset.copy()


# In[115]:


forest_dataset_v2.describe()


# In[116]:


train_forest_dataset, test_forest_dataset = split_dataset(forest_dataset_v2, 0.8)


# In[117]:


train_forest_zscore_dataset, test_forest_zscore_dataset  = Standardization(train_forest_dataset,test_forest_dataset)


# In[118]:


train_forest_zscore_dataset.head()


# In[119]:


test_forest_zscore_dataset.head()


# In[120]:


validated_train_forest, validated_test_forest = cross_validation(forest_dataset_v2,0.8,5)


# In[121]:


validated_test_forest.shape[0]


# In[122]:


validated_train_forest_size = []
for i in range(0, 5):
    validated_train_forest_size.append(validated_train_forest[i].shape[0])
validated_train_forest_size


# In[123]:


forest_accuracy = majority_predictor_regression(validated_train_forest, 'area')


# In[124]:


print("Accuracy: " + str(evaluation_metrics_for_classification_and_regression(forest_accuracy))+ "%")

