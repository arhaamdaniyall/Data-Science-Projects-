#####import data by using pandas
##import numpy for math and numeric:

import pandas as pd

import os
os.chdir("C:\\Users\\ARHAAM DANIYAL SYED\\Desktop\\Developing prediction model of loan risk in banks using data mining\\")

###read csv file
data = pd.read_csv("rawdataset/train.csv")
print("\n\n display data :\n " , data.head())


###information of data:
print("\n\n data info : \n " )
print(data.info())


###display columns names :
print("\n\n columns names : \n" , data.columns)

##data shape (no. of rows and columns):
print("\n\n rows , columns : " , data.shape)

##find null values in data:
print("\n\n display null values : \n" , data.isnull().sum())

###data on columns :
print("\n\n gender : \n" ,data['Gender'].value_counts())

###filling null with mean:
data['Gender']  = data['Gender'].fillna("Male")

###fill values married column:
print("\n\n Married : \n" ,data['Married'].value_counts())
data['Married'] = data['Married'].fillna("Yes")


##fill values in dependent column:
print("\n\n Dependents : \n" ,data['Dependents'].value_counts())
data['Dependents'] = data['Dependents'].fillna(0)

###fillin values in self employed:
print("\n\n Self_Employed : \n" ,data['Self_Employed'].value_counts())
data['Self_Employed'] = data['Self_Employed'].fillna("No")


print("\n\n LoanAmount : \n" ,data['LoanAmount'].value_counts())
data['LoanAmount'] = data['LoanAmount'].fillna(int(data['LoanAmount'].mean()))
print("\n\n LoanAmount : \n" ,data['LoanAmount'].value_counts())


print("\n\n Loan_Amount_Term : \n" ,data['Loan_Amount_Term'].value_counts())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(360.0)

print("\n\n Credit_History : \n" ,data['Credit_History'].value_counts())
data['Credit_History'] = data['Credit_History'].fillna(1.0)


###chck for any null values
print("\n\n check null values : \n" , data.isnull().sum())

data = data.drop(columns=['Loan_ID'])
print(data.columns)



##convert data str to num:
##import label encode
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Gender'] = le.fit_transform(data['Gender'])
data['Married'] = le.fit_transform(data['Married'])
data['Education'] = le.fit_transform(data['Education'])
data['Self_Employed'] = le.fit_transform(data['Self_Employed'])
data['Property_Area'] = le.fit_transform(data['Property_Area'])

###check null values:
print("\n\n null values : \n" , data.isnull().sum())

print(data.info())


##data visualization:
import matplotlib.pyplot as plt
x = data['Loan_Status']
y = data['Education']
plt.bar(x , y)
plt.xlabel("loan status")
plt.ylabel("Education")
plt.show()

###export to csv file:

data['Loan_Status'] = le.fit_transform(data['Loan_Status'])
from pandas import DataFrame

DataFrame(data , columns = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
       'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'])\
    .to_csv("finaldataset/updatedataset.csv", index=False,header=True)

print("\n successfully updated data : ")
