#coding:utf-8

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

def perf_measure(y_actual, list_predict):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(list_predict)): 
        if y_actual[i]==list_predict[i]==1:
            TP += 1
        if list_predict[i]==1 and y_actual[i]==0:
            FP += 1
        if y_actual[i]==list_predict[i]==0:
            TN += 1
        if list_predict[i]==0 and y_actual[i]==0:
            FN += 1

    return (TP, FP, TN, FN)

def main():
    data = pd.read_excel("test.xlsx")
    resMat = data.pop('【瓜】有无病C11')
    data.pop('姓名')
    data.pop('住院号')
    data.pop('数据库编号')
    x_train, x_test, y_train, y_test = train_test_split(data, resMat, test_size=0.3)
    #param = {'criterion':['gini'],'max_depth':[30,50,60,100],'min_samples_leaf':[2,3,5,10],'min_impurity_decrease':[0.1,0.2,0.5]}
    #grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param, cv=6)
    #grid.fit(x_train, y_train)
    #print("最优分类器: ", grid.best_params_, "最优分数: ", grid.best_score_)

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
    clf.fit(x_train, y_train)
    with open("tree.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)

    from sklearn.externals.six import StringIO
    import pydotplus
    
    dot_data = StringIO()
    attr_names = data.columns
    target_name = attr_names
    tree.export_graphviz(clf, out_file=dot_data,feature_names=attr_names,
                         class_names=target_name,filled=True,rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree.png')    
    
    #''''' 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
    #print(clf.feature_importances_)
    
    #'''''测试结果的打印'''  
    #answer = clf.predict(x_test)
    #TP, FP, TN, FN = perf_measure(list(y_test), list(answer))
    #print("TP:%d\tFP:%d\tTN:%d\tFN:%d" %(TP, FP, TN, FN))
    #accuracy = (TP+TN) / (TP+TN+FP+FN)
    #precision = TP / (TP+FP)
    #recall = TP / (TP+FN)
    #specificity = TN / (TN+FP)
    #print("specificity: %f" %specificity)
    #print("recall: %f" %recall)
    #print()

    
    #"""准确率和召回率"""
    #precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
    #answer = clf.predict_proba(data)[:,1]
    #print(classification_report(resMat, answer, target_names=['yes', 'no']))

if __name__ == "__main__":
    for i in range(10):
        main()
    #print("hello")
