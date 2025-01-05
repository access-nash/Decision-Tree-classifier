# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 23:19:52 2025

@author: avina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_hc = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Foundational ML Algorithms I/Healthcare_Dataset_Preprocessed.csv.csv')
df_hc.columns
df_hc.dtypes
df_hc.shape
df_hc.head()

missing_values = df_hc.isnull().sum()
print(missing_values)

for col in ['Diet_Type_Vegan', 'Diet_Type_Vegetarian', 'Blood_Group_AB', 'Blood_Group_B','Blood_Group_O']: 
    if df_hc[col].dtype == 'bool':
        df_hc[col] = df_hc[col].astype(int)
        

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Define features (X) and target (y)
X = df_hc.drop(columns=["Target"])
y = df_hc["Target"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


# Initialize and train the decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Visualizing the decision tree
from sklearn import tree
fig = plt.figure(figsize=(45,15))
feature_names = list(X.columns)
_ = tree.plot_tree(decision_tree, 
                   feature_names=feature_names,  
                   class_names=['Healthy', 'Unhealthy'],
                   filled=True)

# Make predictions on the test set
y_pred = decision_tree.predict(X_test)

# Evaluate the model
classification_report_output = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

# Output results
print("Classification Report:\n", classification_report_output)
print("Accuracy:", accuracy)

# Confusion matrix for visual evaluation
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Unhealthy'], yticklabels=['Healthy', 'Unhealthy'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

feature_imp = decision_tree.feature_importances_
 #Sorting the features by importance in descending order in a dataframe.
imp = pd.DataFrame({'Col_names': X_train.columns, 'Importance': np.round(feature_imp, 2)})\
.sort_values(by='Importance', ascending=False)
imp['cum_imp'] = imp.Importance.cumsum()
imp

drop_col = imp[imp.cum_imp >  0.90]['Col_names'].to_list()
drop_col

# Dropping the columns from X_train
X_train.drop(columns = drop_col, axis = 1, inplace = True)

# Dropping the columns from X_test
X_test.drop(columns = drop_col, axis = 1, inplace = True)

# Building the model again using the modified X_train
DT_model = DecisionTreeClassifier(random_state = 42)
DT_model.fit(X_train, y_train)

# Visualizing the decision tree
from sklearn import tree
fig = plt.figure(figsize=(60,15))
feature_names = list(X.columns)
_ = tree.plot_tree(DT_model, 
                   feature_names=feature_names,  
                   class_names=['Healthy', 'Unhealthy'],
                   filled=True)
# Making the predictions using the modified X_test
y_pred = DT_model.predict(X_test)

#Classification report on train data
class_report_test = classification_report(y_test,y_pred)
print ("The train report is:")
print (class_report_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix for visual evaluation after selecting important features
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Unhealthy'], yticklabels=['Healthy', 'Unhealthy'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Hyperparameter Tuning for Model Optimization
depth = DT_model.get_depth()
# List of max_depth values 
max_depth_list = list(range(depth,0,-3))

from sklearn.metrics import f1_score

# Dictionary to store the train and test F1 scores

train_scores = {}
test_scores = {}

# Loop through max_depth values and train the models
for depth in max_depth_list:
    # Initialize the Decision Tree model with the current max_depth value
    DT_model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    
    # Train the model
    DT_model.fit(X_train, y_train)
    
    # Make predictions on the train dataset
    y_train_pred = DT_model.predict(X_train)

    # Make predictions on the test dataset
    y_test_pred = DT_model.predict(X_test)
    
    # Store the train F1 score in the dictionary with the max_depth as the key
    train_scores[depth] = f1_score(y_train, y_train_pred)
    
    # Store the test F1 score in the dictionary with the max_depth as the key
    test_scores[depth] = f1_score(y_test, y_test_pred)
    

# Print the train and test F1 scores for each model
for depth in max_depth_list:
    print(f"max_depth = {depth}|\
    Train Score = {train_scores[depth]:.3f} |\
    Test score = {test_scores[depth]:.3f}")
    print('_'*65)
    
# Dictionary to store the train and test F1 scores

train_scores = {}
test_scores = {}
accuracy_scores = []

min_sample_leaf_list = list(range(1, 21))
depth = 6

from sklearn.model_selection import train_test_split, KFold, cross_val_score
# Loop through min_sample_leaf values and train the models
for min_sample_leaf in min_sample_leaf_list:
    
    # Initialize the Decision Tree model with the current min_samples_leaf value
    DT_model = DecisionTreeClassifier(min_samples_leaf=min_sample_leaf, max_depth= depth, random_state=42)
    
    # Train the model
    DT_model.fit(X_train, y_train)
    
    # Make predictions on the train dataset
    y_train_pred = DT_model.predict(X_train)

    # Make predictions on the test dataset
    y_test_pred = DT_model.predict(X_test)
    
    # Store the train F1 score in the dictionary with the min_sample_leaf as the key
    train_scores[min_sample_leaf] = f1_score(y_train, y_train_pred)
    
    # Store the test F1 score in the dictionary with the min_sample_leaf as the key
    test_scores[min_sample_leaf] = f1_score (y_test, y_test_pred)
    
    # Perform cross-validation to evaluate performance
    scores = cross_val_score(DT_model, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_scores.append(scores.mean())
    

# Print the train and test F1 scores for each model
for min_sample_leaf in min_sample_leaf_list:
    print(f"min_sample_leaf = {min_sample_leaf}|\
    Train Score = {train_scores[min_sample_leaf]:.3f} |\
    Test Score = {test_scores[min_sample_leaf]:.3f}")
    print('_'*65)
    
# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(min_sample_leaf_list, accuracy_scores, marker='o')
plt.title('Hyperparameter Tuning for min_samples_leaf')
plt.xlabel('min_samples_leaf')
plt.ylabel('Cross-validated Accuracy')
plt.grid()
plt.show()

# Select the best min_samples_leaf
best_min_samples_leaf = min_sample_leaf_list[np.argmax(accuracy_scores)]
print(f"Best min_samples_leaf: {best_min_samples_leaf}")