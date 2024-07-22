# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import OrdinalEncoder, StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from imblearn.over_sampling import SMOTE
# import joblib

# # Load the dataset
# df = pd.read_csv('FastagFraudDetection.csv')

# # Convert Timestamp to datetime format and extract relevant features
# df['Timestamp'] = pd.to_datetime(df['Timestamp'])
# df['Hour'] = df['Timestamp'].dt.hour
# df['Day'] = df['Timestamp'].dt.day
# df['Month'] = df['Timestamp'].dt.month
# df['Weekday'] = df['Timestamp'].dt.weekday

# # Feature: Difference between Transaction_Amount and Amount_paid
# df['Amount_Difference'] = df['Transaction_Amount'] - df['Amount_paid']

# # Ordinal encoding for Vehicle_Dimensions
# dimensions_order = ['Small', 'Medium', 'Large']
# ordinal_encoder = OrdinalEncoder(categories=[dimensions_order])
# df['Vehicle_Dimensions_Encoded'] = ordinal_encoder.fit_transform(df[['Vehicle_Dimensions']])

# # Ordinal encoding for Vehicle_Dimensions
# dimensions_order = ['Bus ','Car','Motorcycle', 'Truck', 'Van' ,'Sedan', 'SUV']
# ordinal_encoder = OrdinalEncoder(categories=[dimensions_order])
# df['Vehicle_Type_encoded'] = ordinal_encoder.fit_transform(df[['Vehicle_Type']])

# # Ordinal encoding for Vehicle_Dimensions
# dimensions_order = ['Regular', 'Express']
# ordinal_encoder = OrdinalEncoder(categories=[dimensions_order])
# df['Lane_Type_encoded'] = ordinal_encoder.fit_transform(df[['Lane_Type']])

# # Drop the original Vehicle_Dimensions column
# df.drop(columns=['Vehicle_Dimensions', 'Geographical_Location'], inplace=True)

# # Encode categorical variables
# df = pd.get_dummies(df, columns=['Vehicle_Type', 'Lane_Type'], drop_first=True)

# # Drop irrelevant or redundant columns
# df.drop(columns=['Transaction_ID', 'Timestamp', 'FastagID', 'TollBoothID', 'Vehicle_Plate_Number'], inplace=True)

# # Define features and target variable
# X = df[['Vehicle_Dimensions_Encoded','Hour','Day','Month','Weekday','Amount_Difference','Vehicle_Type_encoded','Lane_Type_encoded']]
# y = df['Fraud_indicator']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # Standardize the features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Initialize and train the model
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# # Apply SMOTE to the training data
# smote = SMOTE(random_state=42)
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# # Train the model on the resampled data
# model.fit(X_train_resampled, y_train_resampled)

# # Make predictions
# y_pred_resampled = model.predict(X_test)

# # Evaluate the model
# print(confusion_matrix(y_test, y_pred_resampled))
# print(classification_report(y_test, y_pred_resampled))
# print(f'Accuracy: {accuracy_score(y_test, y_pred_resampled)}')

# # Save the model and scaler to files
# joblib.dump(model, 'random_forest_model.joblib')
# joblib.dump(scaler, 'scaler.joblib')

# print("Model and scaler saved to random_forest_model.joblib and scaler.joblib respectively")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('FastagFraudDetection.csv')

# Convert Timestamp to datetime format and extract relevant features
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Hour'] = df['Timestamp'].dt.hour
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Weekday'] = df['Timestamp'].dt.weekday

# Feature: Difference between Transaction_Amount and Amount_paid
df['Amount_Difference'] = df['Transaction_Amount'] - df['Amount_paid']

# Ordinal encoding for categorical features
ordinal_encoder = OrdinalEncoder(categories=[['Small', 'Medium', 'Large'], ['Bus ', 'Car', 'Motorcycle', 'Truck', 'Van', 'Sedan', 'SUV'], ['Regular', 'Express']])
df[['Vehicle_Dimensions_Encoded', 'Vehicle_Type_encoded', 'Lane_Type_encoded']] = ordinal_encoder.fit_transform(df[['Vehicle_Dimensions', 'Vehicle_Type', 'Lane_Type']])

# Drop the original Vehicle_Dimensions and Geographical_Location columns
df.drop(columns=['Vehicle_Dimensions', 'Geographical_Location'], inplace=True)

# Drop irrelevant or redundant columns
df.drop(columns=['Transaction_ID', 'Timestamp', 'FastagID', 'TollBoothID', 'Vehicle_Plate_Number'], inplace=True)

# Encode the target variable
label_encoder = LabelEncoder()
df['Fraud_indicator'] = label_encoder.fit_transform(df['Fraud_indicator'])

# Define features and target variable
X = df[['Vehicle_Dimensions_Encoded', 'Hour', 'Day', 'Month', 'Weekday', 'Amount_Difference', 'Vehicle_Type_encoded', 'Lane_Type_encoded']]
y = df['Fraud_indicator']

# Check class distribution
print("Class distribution:\n", y.value_counts())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=cv, scoring='accuracy')
print(f'Cross-Validation Accuracy: {cv_scores.mean()}')

# Feature Importance
feature_importances = model.feature_importances_
features = ['Vehicle_Dimensions_Encoded', 'Hour', 'Day', 'Month', 'Weekday', 'Amount_Difference', 'Vehicle_Type_encoded', 'Lane_Type_encoded']
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print("Feature Importances:\n", importance_df.sort_values(by='Importance', ascending=False))

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Precision: {precision_score(y_test, y_pred, pos_label=1)}')
print(f'Recall: {recall_score(y_test, y_pred, pos_label=1)}')
print(f'F1-Score: {f1_score(y_test, y_pred, pos_label=1)}')

# Save the model and scaler to files
joblib.dump(model, 'random_forest_model.joblib')
joblib.dump(scaler, 'scaler.joblib')

print("Model and scaler saved to random_forest_model.joblib and scaler.joblib respectively")
