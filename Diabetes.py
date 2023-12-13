from cgi import test
from pyexpat import features
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
#First get some information about the breast cancer dataset
# The best way to convert it to the Dataframe for easy processing.
# First load the dataset as
diabetesData = load_breast_cancer()
#Information about the dataset
print(diabetesData)
#diabetesData contains both the data and the labels (target)
#You can separate the data and labels using the following statments
X = diabetesData['data']
y = diabetesData['target']
#print the feature names
print(diabetesData.feature_names)
#print the target names
print(diabetesData.target_names)
#Convert X,y into dataframe
X = pd.DataFrame(X)
y = pd.DataFrame(y)
#print first 5 rows of X and y to see what it contains?
print(X.head())
print(y.head())
#print the shape of X and y i.e how many rows and columns
print(X.shape)
print(y.shape)
#print specific rows and columns from the datasframe
print(X.iloc[3:6, 1:5]) 
#print information about dataframe X
print(X.info())
#count number of empty cells or null values columnwise
print(f"Nmber of null values in each column are {(X.isnull()).sum()}")
#print number of null values in whole dataset
print(f"Nmber of null values in whole dataset are {((X.isnull()).sum()).sum()}")
#replacing the null values with some predefined values
print(X1.columns) #print values of X
#print(y) #print values of y
#Fill all null entries with 0
print(X.fillna(0))
#Splitting the dataset into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(X,y)#, test_size=0.15, random_state=1)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
#Applying Decision Tree Model
decTreeModel = DecisionTreeClassifier()
decTreeModel.fit(X_train,y_train)
y_predictions = decTreeModel.predict(X_test)
print(f"Accuracy of Decision Tree model is {accuracy_score(y_predictions,y_test)}")