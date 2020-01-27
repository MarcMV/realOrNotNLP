#-----------------------------------------------------------------------

#NLP: NLP - or Natural Language Processing - is shorthand for a wide array of techniques designed to help machines learn from text. Natural Language Processing powers everything from chatbots to search engines, and is used in diverse tasks like sentiment analysis and machine translation.

#Challenge:
#-Diferenciate between metaphoric uses of the language vs. real emergencies. 
#-Predict what tweets are about real disasters VS NOT. 
#-Dataset of 10'000 tweets that are hand classified. 

#Evaluation:
#-Submissions are evaluated using the F1 between the predicted and expected results

#Submission:
#-For each ID in the test, you must predict 1 if the tweet is describing a real disaster and 0 otherwise. 

#-----------------------------------------------------------------------

import numpy as np
import pandas as pd 
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
import os

#------STEP 1: Setup & Exploration------

#Find where my file is
print(os.getcwd())
print(os.listdir("./Projects/Real or not (NLP)/nlp-getting-started"))


#Load data
train_df = pd.read_csv("./Projects/Real or not (NLP)/nlp-getting-started/train.csv")
test_df = pd.read_csv("./Projects/Real or not (NLP)/nlp-getting-started/test.csv")

#Check success
train_df.head()

#Examples of false emergencies
train_df[train_df["target"] == 0]["text"].head().to_frame()

#Examples of false emergencies
train_df[train_df["target"] == 1]["text"].head().to_frame()

#Explore a couple in detail
train_df[train_df["target"] == 1]['text'][0]
train_df[train_df["target"] == 1]['text'][1]
train_df[train_df["target"] == 1]['text'][2]

#------STEP 2: Vectorizing tweets------

#Building vectors... The theory behind the model we'll build in this notebook is pretty simple: the words contained in each tweet are a good indicator of whether they're about a real disaster or not (this is not entirely correct, but it's a great place to start).

#We'll use scikit-learn's CountVectorizer to count the words in each tweet and turn them into data our machine learning model can process.

#Init vectorizer
count_vectorizer = feature_extraction.text.CountVectorizer()

#Example to check how it works, counts for the first few tweets
example_train_vect = count_vectorizer.fit_transform(train_df["text"][0:6])
print(example_train_vect[0].todense().shape) #we use to dense, as the vectors are "sparse", i.e. only non zero elements are kept to save space
print(example_train_vect[0].todense())

#Vectorize tweets
train_vectors = count_vectorizer.fit_transform(train_df["text"])
test_vectors = count_vectorizer.transform(test_df['text']) #Note we're using transform here, not fit_transform, this makes sure that the tokens in the train vectors are the only ones mapped to the test vectors. i.e. that train and test vectors use the same set of tokens. 


#------STEP 3: Model------

#In this model, we're assuming words in each tweet are a good indicator of whether they're about a real disaster or not. The presence of particular word (or set of words) in a tweet might link directly to whether or not that tweet is real. What we're assuming here is a linear connection. 

#Init model
clf = linear_model.RidgeClassifier() #The vectors are very big, we want to push weights toward 0 without disccounting different words. Ridge regression does this well. 

#Train & get scores using cross validation
model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")

#There are lots of ways to potentially improve on this (TFIDF, LSA, LSTM / RNNs, the list is long!) - give any of them a shot!

#------STEP 4: Submit------

#Train model
clf.fit(train_vectors, train_df["target"])

#Load submission document
sample_submission = pd.read_csv("./Projects/Real or not (NLP)/nlp-getting-started/sample_submission.csv")
print(sample_submission)

#Add model predictions
sample_submission['target'] = clf.predict(test_vectors)

sample_submission.head()

#Save to csv and submit for competition
sample_submission.to_csv("submission.csv", index=False)