#coding:utf-8

#from sklearn.decomposition import PCA
#import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

data = pd.read_excel("test.xlsx")
resMat = data.pop('【瓜】有无病')
data.pop('姓名')

row = data.shape[0]
col = data.columns
my_imputer = Imputer()
data_imputed = my_imputer.fit_transform(data)
print(type(data_imputed))
print(data_imputed)

print("hello")