import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#sample emails
emails = [
    "Free money now",
    "Win a free lottery",
    "Hello friend, how are you?", 
    "Meeting at noon",
    "Win money now"
    ]

labels = [1, 1, 0, 0, 1]  #1 - spam, 0 - not spam

#convert the text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)

print(X.toarray())

#splitting the data into train and test 
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

#Train the NB classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)


new_emails = [
    "Free trip Jamaica",
    "Win a lottery",
    "Your friend called money",
    "AI take jobs",
    "AI not jobs"
]

X_new = vectorizer.fit_transform(new_emails)


predictions = classifier.predict(X_new)

print("Predictions:", predictions)

#calculate accuracy
#accuracy = accuracy_score(y_test, y_pred)

#print(f'Accuracy: {accuracy * 100:.2f}%')