import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#create the dataset
data = {
    'Hours Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Pass': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

x = df[['Hours Studied']]

y = df['Pass']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

log_reg_model = LogisticRegression()

#fit function is training the model
log_reg_model.fit(x_train, y_train)

predictions = log_reg_model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix: {conf_matrix}')
print(f'Predictions: {predictions}')