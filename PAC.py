# -------------------------------------------------------------------------
# AUTHOR: Ajith Elumalai
# FILENAME: pca.py
# SPECIFICATION: Perform PCA multiple times on the heart disease dataset, removing one feature at a time.
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: 1 day
# -------------------------------------------------------------------------

# Importing necessary Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("heart_disease_dataset.csv")

# Create a training matrix 
df_features = df.iloc[:, :-1]  # Drop the last column (target variable)

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

# Get the number of features
num_features = df_features.shape[1]

# Store PC1 variances and corresponding removed features
pc1_variances = []

# Run PCA for each feature removed
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    reduced_data = np.delete(scaled_data, i, axis=1)
    
    # Run PCA on the reduced dataset
    pca = PCA()
    pca.fit(reduced_data)
    
    # Store PC1 variance and the feature removed
    pc1_variances.append((pca.explained_variance_ratio_[0], df_features.columns[i]))

# Find the maximum PC1 variance and the feature removed
max_variance, removed_feature = max(pc1_variances, key=lambda x: x[0])

# Print results
print(f"Highest PC1 variance found: {max_variance:.4f} when removing {removed_feature}")
