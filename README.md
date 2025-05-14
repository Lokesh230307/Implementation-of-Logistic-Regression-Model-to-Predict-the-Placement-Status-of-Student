# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load & Preprocess: Read placement CSV, drop "sl_no", "salary", encode categorical columns.
2. Split Data: Extract features (X) and target (y), split into train/test (20% test).
3. Train & Predict: Train Logistic Regression model, predict test set outcomes.
4. Evaluate: Compute accuracy, confusion matrix, print classification report, predict sample.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
data=pd.read_csv('/content/Placement_Data.csv')
data.head()
data1=data.copy()
data1 = data1.drop(["sl_no", "salary"], axis = 1)#removes the specified row or column
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") # A Library for Large Linear Classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred) #Accuracy Score = (TP+TN)/ (TP+FN+TN+FP) , True
#accuracy_score(y_true, y_pred, normalize=False)
#Normalize : It contains the boolean value(True/False).If False, return the number of cor
#Otherwise, it returns the fraction of correctly confidential samples.
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion #11+24=35 -correct predictions, 5+3=8 incorrect predictions
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)
![Screenshot 2025-05-14 105607](https://github.com/user-attachments/assets/4fef0d8c-fa44-4b4d-b5d8-2517126a9f99)
![Screenshot 2025-05-14 105636](https://github.com/user-attachments/assets/adea2430-4eee-49d8-ba49-2e38be7749b0)
![Screenshot 2025-05-14 105701](https://github.com/user-attachments/assets/143f3637-d089-426f-acf9-961d81b06ea0)
![Screenshot 2025-05-14 105738](https://github.com/user-attachments/assets/eed71902-0d08-4f79-970f-0a9108245679)
![Screenshot 2025-05-14 105824](https://github.com/user-attachments/assets/0423d612-24d3-4bc0-b018-18438e21969a)
![Screenshot 2025-05-14 105851](https://github.com/user-attachments/assets/55d5af7a-2c6f-4372-b9b3-8f3779f744f2)

![Screenshot 2025-05-14 105924](https://github.com/user-attachments/assets/618c75db-1b1e-4e80-a99d-567a848547f6)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
