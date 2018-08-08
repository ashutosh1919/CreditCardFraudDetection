import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import model_selection
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


data = pd.read_csv('creditcard.csv')
data = data.sample(frac = 0.1, random_state = 1)

Fraud = data[data['Class']==1]
Valid = data[data['Class']==0]
outlier_fraction = len(Fraud)/float(len(Valid))

# Number of frauds in training.
print(outlier_fraction)
print('Fraud Cases : {}'.format(len(Fraud)))
print('Valid Cases : {}'.format(len(Valid)))

# Correlation matrix
cormat = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(cormat,vmax=0.8,square=True)
plt.show()

# Get all the colums from DataFrame
columns = data.columns.tolist()

columns = [c for c in columns if c not in ['Class']]

# Store the variable we will be predicting on.
target = 'Class'

# Data to be analysed.
X = data[columns]
Y = data[target]

# Define random state and validation size
state = 1
validation_size = 0.5

#Split the data into train and test
X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size=validation_size,random_state=state)

# Define the outlier detection methods
classifiers = {
    "Isolation Forest" : IsolationForest(max_samples=len(X),contamination=outlier_fraction,random_state=state),
    "Local Outlier Factor" : LocalOutlierFactor(n_neighbors=20,contamination=outlier_fraction)
}

# Fit the model.
n_outliers = len(Fraud)

for i,(clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag outlier.
    if(clf_name == "Local Outlier Factor"):
        #y_pred = clf.fit_predict(X)
        clf.fit(X_train,Y_train)
        y_pred = clf.fit_predict(X_test)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X_train)
        scores_pred = clf.decision_function(X_test)
        y_pred = clf.predict(X_test)
        y_pred_train = clf.predict(X_train)

    # Reshape the prediction values to 0 for valid, 1 for fraud.
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y_test).sum()

    # Run Classification metrics
    print('{} : {}'.format(clf_name,n_errors))
    print(accuracy_score(Y_test,y_pred))
    print(classification_report(Y_test,y_pred))

