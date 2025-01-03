# Data Handling
import pandas as pd
import numpy as np
import math

# Machine Learning Tools
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb

output_text = ""

data = pd.read_csv("/Users/george/Documents/GitHub/Machine_Learning_Rainfall_Prediction/usa_rain_prediction_dataset_2024_2025.csv")

data = data.drop(columns=["Date", "Location"])

# Remove whitespace to accommodate for LightGBM
data.columns = data.columns.str.replace(' ', '_')

X = data.drop(columns=["Rain_Tomorrow"])

y = data["Rain_Tomorrow"]



# INITIAL MODEL USING LIGHTGBM



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to LightGBM format
train_data = lgb.Dataset(X_train, label = y_train)
test_data = lgb.Dataset(X_test, label = y_test, reference = train_data)

params = {
    'objective': 'binary',           # Specify model is for binary classification
    'metric': 'binary_logloss',      # Evaluation metric - how well predictions match true labels
    'boosting_type': 'gbdt',         # "gdbt" = Gradient boosting decision tree
    'num_leaves': 12,                # Maximum leaves in one tree
    'verbosity': -1
}

# Train LightGBM model
lgb_model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, test_data],                 # Evaluate on both train and test data
    num_boost_round=100,                                # Number of boosting iterations
    callbacks=[
        lgb.early_stopping(stopping_rounds=10),         # Stop if no improvement in 10 rounds
        lgb.early_stopping(10, verbose=0), lgb.log_evaluation(period=0)
        ],           
)

# Predict probabilities
y_pred_prob = lgb_model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

output_text += f"\n\n--Initial LightGBM:\nAccuracy: {accuracy}\n\n"

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)
output_text += f"Confusion Matrix:\n{cm}\n\n"

# Find most influential feature (Most gain)
importance = lgb_model.feature_importance(importance_type='gain')
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
    }).sort_values(by='Importance', ascending = False)

most_influential_feature = feature_importance_df.iloc[0]

output_text += f"Most Influential Feature: {most_influential_feature['Feature']}\n"
output_text += f"Importance Score: {most_influential_feature['Importance']}\n\n"


# K FOLD CROSS-VALIDATION ON LIGHTGBM


# KFold with 5 splits
kf = KFold (n_splits=5, shuffle=True, random_state=42)

#Empty lists to store accuracies and confusion matrices
accuracies = []
confusion_matrices = []

#Loop through each fold
for train_index, val_index in kf.split(X):
    
    #Split data into training and validation sets
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Create LightGBM datasets, with parameters
    train_data = lgb.Dataset(X_train, label = y_train)
    val_data = lgb.Dataset(X_val, label = y_val)

    params = {
        'objective': 'binary',      
        'metric': 'binary_logloss',     
        'boosting_type': 'gbdt',         
        'num_leaves': 10,         
        'verbosity': -1
    }
    
    # Train model
    lgb_model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],   
        num_boost_round=100,      
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),         # Stop if no improvement in 10 rounds
            lgb.early_stopping(10, verbose=0), lgb.log_evaluation(period=0)
            ],
    )
    
    # Predict on validation data
    y_pred_prob = lgb_model.predict(X_val)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    
    #Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)
    
    #Calculate confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    confusion_matrices.append(cm)

#Find mean accuracy
average_accuracy = np.mean(accuracies)

#Aggregate confusion matrices
overall_cm = np.sum(confusion_matrices, axis=0)

output_text += f'\n\n--LightGBM 5-Fold Cross-Validation:\nAverage Accuracy: {average_accuracy:.4f}\n\nConfusion Matrix:\n{overall_cm}\n\n'


# K-NEAREST NEIGHBOR MODEL WITH 5 FOLD CROSS VALIDATION


kf = KFold(n_splits=5, shuffle=True, random_state= 42)
accuracies = []
confusion_matrices = []

n_neighbors = int(math.sqrt(len(X)))

for train_index, val_index in kf.split(X):
    
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = KNeighborsClassifier(n_neighbors = n_neighbors)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    accuracies.append(accuracy)
    
    cm = confusion_matrix(y_val, y_pred)
    confusion_matrices.append(cm)
    
average_accuracy = np.mean(accuracies)
overall_cm = np.sum(confusion_matrices, axis=0)

output_text += f'\n\n--KNN With 5-Fold Cross-Validation:\nAverage Accuracy: {average_accuracy:.4f}\n\nConfusion Matrix:\n{overall_cm}\n\n'

#print(y.value_counts())
print(output_text)

