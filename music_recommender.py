import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load data
music_data= pd.read_csv('music.csv')
music_data
X=music_data.drop(columns=['genre'])
y=music_data['genre']

#model
model= DecisionTreeClassifier()
model.fit(X,y)
predictions= model.predict([[21,1],[22,0]])
predictions

#for calculating accuracy

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)

#model
model= DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions= model.predict(X_test)
predictions

#accuracy
score=accuracy_score(y_test, predictions)
score