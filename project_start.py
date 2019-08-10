#coding:utf-8

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

data = pd.read_excel("test.xlsx")
resMat = data.pop('【瓜】有无病')
data.pop('姓名')
pca = PCA(n_components=4)

pca.fit(data)
print(pca.explained_variance_ratio_)
print("hello")

#from sklearn import tree
#from sklearn.preprocessing import LabelEncoder
#import numpy as np
#import pandas as pd
#import os
#os.environ["PATH"] += os.pathsep + 'D:\program file\Graphviz2.38\bin'

#attr_arr=[['slashdot','USA','yes',18,'None'],
         #['google','France','yes',23,'Premium'],
         #['digg','USA','yes',24,'Basic'],
         #['kiwitobes','France','yes',23,'Basic'],
         #['google','UK','no',21,'Premium'],
         #['(direct)','New Zealand','no',12,'None'],
         #['(direct)','UK','no',21,'Basic'],
         #['google','USA','no',24,'Premium'],
         #['slashdot','France','yes',19,'None'],
         #['digg','USA','no',18,'None'],
         #['google','UK','no',18,'None'],
         #['kiwitobes','UK','no',19,'None'],
         #['digg','New Zealand','yes',12,'Basic'],
         #['slashdot','UK','no',21,'None'],
         #['google','UK','yes',18,'Basic'],
         #['kiwitobes','France','yes',19,'Basic']]

#dataMat = np.mat(attr_arr)
#arrMat  = dataMat[:, 0:4]   #属性数据值
#resMat  = dataMat[:, 4]     #属性结果值
#attr_names = ['src', 'address', 'FAQ', 'num']   #属性名称
#attr_pd = pd.DataFrame(data=arrMat, columns=attr_names)
#print(attr_pd)

#le = LabelEncoder()
#for col in attr_pd.columns:
    #attr_pd[col] = le.fit_transform(attr_pd[col])   #为每一列序列化,就是将每种字符串转化为对应的数字。用数字代表类别
##print(attr_pd)

#clf = tree.DecisionTreeClassifier()
#clf.fit(attr_pd, resMat)

#result = clf.predict([[1,1,1,0]])
#print(result)

#from sklearn.externals.six import StringIO
#import pydotplus

#dot_data = StringIO()
#target_name=['None','Basic','Premium']
#tree.export_graphviz(clf, out_file=dot_data,feature_names=attr_names,
                     #class_names=target_name,filled=True,rounded=True,
                     #special_characters=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_png('D:/tree.png')

