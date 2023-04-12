#import the dependencies

#used for numpy arrays
import numpy as np

#used for creating data frames
import pandas as pd

#to standardise the function : a standardiser
from sklearn.preprocessing import StandardScaler

#to split data in training data and testing data
from sklearn.model_selection import train_test_split

#import support vector machine
from sklearn import svm
from sklearn.metrics import accuracy_score

# download file from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download
#to know more about a function
#pd.read_csv?

#loading diabates dataset to pandas dataframes
diabetes_data = pd.read_csv('/content/diabetes.csv')

#printing the first five rows of the dataset
diabetes_data.head()

#number of rows and columns in this dataset
diabetes_data.shape

#getting the statistical aspects of dataset ie mean,min,max etc
diabetes_data.describe()

diabetes_data['Outcome'].value_counts()

#grouping data on basis of Outcome column and finding mean values of each column
diabetes_data.groupby('Outcome').mean()
#seperating the data and labels : excluding the outcome column
#axis = 1 if deopping col, and 1 if dropping row
X = diabetes_data.drop(columns = 'Outcome', axis = 1)
Y = diabetes_data['Outcome']

print(X)
print(Y)

#data standardisation
scaler = StandardScaler()
scaler.fit(X)
standardised_data = scaler.transform(X)
print(standardised_data)
X= standardised_data

#train test split
X_train, X_test, Y_train, Y_split = train_test_split(X,Y, test_size =0.2, stratify=Y, random_state= 2)


#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2, stratify=Y, random_state= 2)
#stratifying is used to divide the data in proportion, so that not all diabaetic goes to one data frame and otherwise

#training the model
classifier = svm.SVC(kernel = 'linear')
#training the support vector machine classifier
classifier.fit(X_train, Y_train)

#model evaluation
#accuracy score on training data
X_train_predicton = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predicton, Y_train)
print(training_data_accuracy)

#model evaluation
#accuracy score on test data
X_test_predicton = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predicton, Y_test)
print(test_data_accuracy)

#making a predictive system

input_data= (0,111,40,35,168,43.1,2.288,33)

#changing input data to numpy array
input_data_asnp_array = np.asarray(input_data)

#reshape the data for one instance
input_data_reshaped = input_data_asnp_array.reshape(1,-1)

#standardise the data
std_data = scaler.transform(input_data_reshaped)

print(std_data)
prediction = classifier.predict(std_data)
print(prediction)
