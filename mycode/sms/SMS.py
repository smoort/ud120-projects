# Import Pandas library
import pandas as pd   

# Dataset from https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
df = pd.read_csv('smsspamcollection/SMSSpamCollection', 
                   sep='\t',
                   names=['label','sms_message'])

# Output printing out first 5 rows
print(df.head())

# Conversion
df['label'] = df.label.map({'ham':0, 'spam':1})
print(df.head())

# Print dataset shape
print(df.shape)

# split into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], df['label'], random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# Import the count vectorizer and initialize it
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

# Training
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

# Prediction
predictions = naive_bayes.predict(testing_data)
print(predictions)

# Measure model performance

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))