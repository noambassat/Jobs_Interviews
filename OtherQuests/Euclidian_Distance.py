#
# !pip install scikit-learn
# !pip install imbalanced-learn

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Prepare your dataset with the features (X) and labels (y)

# Assuming X and y are already prepared


X = [[1,2,3],[5,6,7],[1,7,3],[12,3,7],[11,3,8],[23,5,9],[6,8,3],[55,44,12],[45,2,7],[4,2,7],[20,7,3],[79,4,6],[23,44,92],[9,2,9],[7,2,7],[6,6,6]]
y = [1,1,1,2,2,2,0,0,0,0,0,0,0,0,0,0]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(y_train)
# Apply SMOTE to oversample the small classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(X_train_resampled)
print(y_train_resampled)
# Initialize the random forest classifier as the base estimator
base_estimator = RandomForestClassifier()

# Initialize the AdaBoost classifier with the random forest as the base estimator
boosted_rf = AdaBoostClassifier(base_estimator=base_estimator, random_state=42)

# Train the boosted random forest on the resampled training data
boosted_rf.fit(X_train_resampled, y_train_resampled)

# Make predictions on the testing data
y_pred = boosted_rf.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))


#
# In this updated code, we import the SMOTE class from the imblearn.over_sampling module. After splitting the dataset into training and testing sets, we apply the fit_resample method of SMOTE to oversample the small classes in the training data. The resulting resampled training set, X_train_resampled and y_train_resampled, is then used for training the boosted random forest classifier.
#
# The remaining parts of the code, such as initializing the random forest and AdaBoost classifiers, making predictions, and printing the classification report, remain the same as before.
#
# Make sure you have the imbalanced-learn library installed (pip install imbalanced-learn) to use the SMOTE class.
#
# Adjust the code according to your specific dataset and problem requirements, including any necessary preprocessing steps or modifications.