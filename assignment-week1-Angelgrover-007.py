import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder

#Q1.
df=pd.read_csv('Dataset 2.csv')
df.head()

#Q2.
print('Shape: ',df.shape)

#Q3.
print('Column Names')
print(df.columns)

#Q4.
print('\nData Types')
print(df.dtypes)

#Q5
print('Missing value:',df.isnull().sum())

#Q6
print('Average Age:',df['Age'].mean())

#Q7
print('Average watch hours per week:',df['WatchHoursPerWeek'].mean())

#Q8
print('Average monthly spending of users:',df['MonthlySpend'].mean())

#Q9
#d
print('Number of users in each subscription category:')
print(df['SubscriptionType'].value_counts())

#Q10
#d
print('The percentage of users who renewed their subscriptions:')
print((df['SubscriptionRenewed']=='Yes').mean()*100)

#Q11
le = LabelEncoder()
for col in ['Gender','SubscriptionType','FavoriteGenre','SubscriptionRenewed']:
    df[col] = le.fit_transform(df[col])

#Q12
X = df.drop(['SubscriptionRenewed','MonthlySpend'], axis=1)
y = df['SubscriptionRenewed']

#Q13
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Q14
dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_train,y_train)
pred=dt.predict(X_test)

#Q15
print('Decision Tree Accuracy',accuracy_score(y_test,pred))

#Q16
print('Confusion Matrix',confusion_matrix(y_test,pred))

#Q17
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

#Q18
knn_pred=knn.predict(X_test)
print('KNN Accuracy:',accuracy_score(y_test,knn_pred))

#Q19
X_reg = df.drop('MonthlySpend', axis=1)
y_reg = df['MonthlySpend']
X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg,test_size=0.20,random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)

#Q20
new_user = X_reg.iloc[[0]]
prediction = lr.predict(new_user)
print(prediction)

