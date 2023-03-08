

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv('Kaggle_Training_Dataset_v2.csv')
X = dataset.iloc[:40000, [5,6,7,8,9,10,11]].values
y = dataset.iloc[:40000, [21]].values


              
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
X, y = ros.fit_sample(X, y)

                
#missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN' , strategy = 'mean' , axis = 0 )
imputer = imputer.fit(X[ : , 1:8])
X[: , 1:8] = imputer.transform(X[: , 1:8])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting randomforest to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

#from sklearn.metrics import roc_auc_score
#print(roc_auc_score(y_test, y_pred))


