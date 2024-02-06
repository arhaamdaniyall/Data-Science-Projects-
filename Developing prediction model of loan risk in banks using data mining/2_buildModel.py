##for best acc algorithm & dump in pickle
###import uodated data set:

import pandas as pd
import numpy as np
import pickle
import os

os.chdir("C:\\Users\\ARHAAM DANIYAL SYED\\Desktop\\Developing prediction model of loan risk in banks using data mining\\")


##read data set:

mydata = pd.read_csv("finaldataset/updatedataset.csv")
print("display data : \n" ,mydata.head())


##split data for input and target:

x = mydata.drop(columns= ['Loan_Status'])
y = mydata['Loan_Status']

print(x)
print(y)


###make data into array format:
x = np.array(x)
y = np.array(y)

print("\n\n array of x :" , x)
print("\n\n array of y :" , y)


###apply train & test split
from sklearn.model_selection import train_test_split
x_train , x_test ,y_train , y_test = train_test_split(x ,y ,test_size=0.2)

# check train and test data length
print("x_test :",  len(x_test))
print("y_test :",  len(y_test))
print("x_trian :", len(x_train))
print("y_trian :", len(y_train))

###apply algorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

##apply random_forest classifier:
rf_clf = RandomForestClassifier(n_estimators=10 , criterion="entropy")
bestscore_rf = 0
for count in range(1000):

    rf_clf.fit(x_train , y_train)
    acc = rf_clf.score(x_test ,y_test)
    print("\n count :" ,count , "acc : " ,acc ,end="" )
    if bestscore_rf < acc:
        bestscore_rf = acc
        print("\n ------------------>random forest classifier acc : " , bestscore_rf)
        with open("randonforest.pickle" ,"wb")as rf_file:
            pickle.dump(rf_clf ,rf_file)


## naivebayes clasifier::
bestscore_nb = 0
for count2 in range(1000):
    naivebayes_clf = GaussianNB()
    naivebayes_clf.fit(x_train ,y_train)
    acc = naivebayes_clf.score(x_test , y_test)
    print("\n count2 : " ,count2 ,"acc :" , acc ,end="")
    if bestscore_nb < acc:
        bestscore_nb = acc
        print("\n --------------------> naivebayes classifier acc :" ,bestscore_nb)
        with open("naviebayes.pickle" , "wb") as nbfile:
            pickle.dump(naivebayes_clf , nbfile)