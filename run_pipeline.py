import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

# Load dataset
data = pd.read_excel('AmesHousing.xlsx', sheet_name='AmesHousing')

data['Age'] = data['Yr Sold'] - data['Year Built']

# Split into train + val and test sets
test_data = data.sample(n=50, random_state=42)
train_val_data = data.drop(test_data.index)

# Selected Features Setup
selected_features = ['Age', 'Gr Liv Area', 'Lot Area', 'Overall Qual', 'Neighborhood']
target = 'SalePrice'
X_selected = train_val_data[selected_features]
y_selected = train_val_data[target]

# Preprocess Selected Features
preprocessor_selected = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Gr Liv Area', 'Lot Area', 'Overall Qual']),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['Neighborhood'])
    ])
X_selected_processed = preprocessor_selected.fit_transform(X_selected)

# All Features Setup
X_all = train_val_data.drop('SalePrice', axis=1)
y_all = train_val_data['SalePrice']

# Identify numerical and categorical columns
categorical_cols = X_all.select_dtypes(include=['object']).columns
numerical_cols = X_all.select_dtypes(exclude=['object']).columns

# Default values
default_values = {col: (X_all[col].median() if col in numerical_cols else X_all[col].mode().iloc[0]) for col in X_all.columns}
default_all_df = pd.DataFrame([default_values]).reindex(columns=X_all.columns)
default_all_df.to_csv('default_all_features.csv', index=False)

# Preprocessing for all features
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

preprocessor_all = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])
X_all_processed = preprocessor_all.fit_transform(X_all)

# Split data
X_train_selected, X_val_selected, y_train_selected, y_val_selected = train_test_split(
    X_selected_processed, y_selected, test_size=0.2, random_state=42)
X_train_all, X_val_all, y_train_all, y_val_all = train_test_split(
    X_all_processed, y_all, test_size=0.2, random_state=42)

# Set up early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# ANN with Selected Features
model_selected = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train_selected.shape[1],),
                       kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])
model_selected.compile(optimizer='adam', loss='mse')
model_selected.fit(X_train_selected, y_train_selected, validation_data=(X_val_selected, y_val_selected),
                   epochs=200, batch_size=32, callbacks=[early_stop], verbose=0)
model_selected.save('model_selected.h5')

# ANN with All Features
model_all = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train_all.shape[1],),
                       kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dense(1)
])
model_all.compile(optimizer='adam', loss='mse')
model_all.fit(X_train_all, y_train_all, validation_data=(X_val_all, y_val_all),
              epochs=300, batch_size=32, callbacks=[early_stop], verbose=0)
model_all.save('model_all.h5')

# Save preprocessing pipelines
joblib.dump(preprocessor_selected, 'preprocessor_selected.pkl')
joblib.dump(preprocessor_all, 'preprocessor_all.pkl')

print("All files generated successfully!")
