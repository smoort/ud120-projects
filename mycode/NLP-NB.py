# coding: utf-8
# cleaning texts 
import pandas as pd 
import re 
import nltk 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer 

dataset = [["I liked the movie", "positive"],
		["It’s a good movie. Nice story", "positive"],
		["Hero’s acting is bad but heroine looks good. Overall nice movie", "positive"], 
		["Nice songs. But sadly boring ending.", "negative"], 
		["sad movie, boring movie", "negative"]] 
		
dataset = pd.DataFrame(dataset) 
dataset.columns = ["Text", "Reviews"] 

nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 

corpus = [] 

for i in range(0, 5): 
	#print("text = ", dataset['Text'][i])
	text = re.sub('[^a-zA-Z\s]', '', dataset['Text'][i]) 
	#print("after re = ", text)
	text = text.lower() 
	#print("after lower = ", text)
	ps = PorterStemmer()
	text = ps.stem(text)
	#print("after ps = ", text)
	text = text.split() 
	#print("after split = ", text)
	text = [w for w in text if not w in stop_words]
	#print("after stopword = ", text)
	text = " ".join(text)
	#print("after join = ", text)	
	corpus.append(text)
print("corpus = ", corpus)

# creating bag of words model 
cv = CountVectorizer(max_features = 1500) 

X = cv.fit_transform(corpus).toarray() 
print(cv.fit(corpus).vocabulary_)
y = dataset.iloc[:, 1].values
print("X = ", X)


# splitting the data set into training set and test set 
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split( 
		X, y, test_size = 0.25, random_state = 0) 


# fitting naive bayes to the training set 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.metrics import confusion_matrix 

classifier = MultinomialNB(); 
classifier.fit(X_train, y_train) 

# predicting test set results 
y_pred = classifier.predict(X_test) 

# making the confusion matrix 
cm = confusion_matrix(y_test, y_pred) 
print (cm)