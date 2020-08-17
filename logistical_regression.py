import numpy as np
import pandas as pd

dataset = pd.read_csv('./Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values # ignore the rowNumber, CustomerId and Surname
y = dataset.iloc[:, -1].values 

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train 
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier.fit(X_train, y_train)



# Predict
y_pred = classifier.predict(X_test)


# Compute Confusion Metrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Compute Accuracy Score
acc = accuracy_score(y_test, y_pred)

print(acc)
