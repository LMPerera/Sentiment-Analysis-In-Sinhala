import csv

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier

# Read the labeled training data
with open('dataset.csv') as csv_file:
    train = csv.reader(csv_file, delimiter=",", quotechar='"')
    print train
    train.next()
    data = []
    target = []
    for row in train:
        if row[0] and row[1]:
            data.append(row[0])
            target.append(row[1])

# Initialize the "CountVectorizer" object
vectorizer = CountVectorizer(binary='true')

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
data = vectorizer.fit_transform(data)

# Transform a count matrix to a normalized tf-idf representation
# term-frequency times inverse document-frequency (tf-idf)
# tf_idf_data = TfidfTransformer(use_idf=False).fit_transform(data)

# convert the result to an array
data = data.toarray()

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
forest = forest.fit( data, train["sentiment"] )

# Use the random forest to make sentiment label predictions
result = forest.predict(data)

# output
output = pd.DataFrame(data={"id":train["id"], "sentiment":result})