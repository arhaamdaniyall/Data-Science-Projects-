##load pickle file for prediction:

import pickle
import pandas as pd
import numpy as np
import os

os.chdir("C:\\Users\\ARHAAM DANIYAL SYED\\Desktop\\Developing prediction model of loan risk in banks using data mining\\")


data = pd.read_csv("finaldataset/pred.csv")
print("\n\n display data \n",data.head())


inputdata = np.array(data)

pickle_nb = open("naviebayes.pickle" ,"rb")
nb_pickle = pickle.load(pickle_nb)
print("\n\n  prediction values of naive bayes: \n 1: loan approved ; 0 : not approved \n" ,nb_pickle.predict(inputdata))

