from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# Define the number of customers
num_customers = 1000

# Define the classes and their respective sizes
minor_classes = [0, 1, 2]
major_class = 3
minor_class_size = int(num_customers * 0.025)
major_class_size = num_customers - (len(minor_classes) * minor_class_size)

# Create an empty DataFrame to store the customer data
columns = ["age", "income", "credit_score", "class"]
data = pd.DataFrame(columns=columns)

# Generate data for the minor classes
for class_label in minor_classes:
    age = np.random.randint(18, 65, size=minor_class_size)
    income = np.random.randint(20000, 80000, size=minor_class_size)
    credit_score = np.random.randint(300, 850, size=minor_class_size)
    class_data = pd.DataFrame({"age": age, "income": income, "credit_score": credit_score, "class": class_label})
    data = pd.concat([data, class_data], ignore_index=True)

# Generate data for the major class
age = np.random.randint(18, 65, size=major_class_size)
income = np.random.randint(20000, 80000, size=major_class_size)
credit_score = np.random.randint(300, 850, size=major_class_size)
class_data = pd.DataFrame({"age": age, "income": income, "credit_score": credit_score, "class": major_class})
data = pd.concat([data, class_data], ignore_index=True)

# Shuffle the data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate the features (X) and the target variable (y)
X = data.drop("class", axis=1)
y = data["class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a DataFrame with the features and target variable
train_data = pd.concat([X_train, y_train], axis=1)

# Split the training data into subsets for Bagging Random Forest
subset_size = int(len(train_data) * 0.25)  # Size of each subset
n_subsets = len(minor_classes)  # Number of subsets

subsets = []

# Create subsets with oversampled minor classes and random samples from the major class
for _ in range(n_subsets):
    # Oversample minority classes without replacement
    minority_data = train_data[train_data["class"].isin(minor_classes)]
    oversampled_minority = minority_data.groupby("class").apply(lambda x: x.sample(subset_size, replace=False))

    # Randomly sample from the major class
    majority_data = train_data[~train_data["class"].isin(minor_classes)]
    majority_sampled = majority_data.sample(n=subset_size, random_state=42)

    # Combine the oversampled minority and randomly sampled majority data
    subset_data = pd.concat([oversampled_minority, majority_sampled], axis=0)

    subsets.append(subset_data)

# Train individual Random Forest models on each subset
models = []
for subset in subsets:
    X_subset = subset.drop("class", axis=1)
    y_subset = subset["class"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_subset, y_subset)
    models.append(model)

# Make predictions on the test set using each model
predictions = []
for model in models:
    y_pred = model.predict(X_test)
    predictions.append(y_pred)

# Combine the predictions using majority voting
ensemble_predictions = []
for i in range(len(X_test)):
    preds = [predictions[j][i] for j in range(n_subsets)]
    ensemble_pred = max(set(preds), key=preds.count)
    ensemble_predictions.append(ensemble_pred)

# Print classification report
print(classification_report(y_test, ensemble_predictions))