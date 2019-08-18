#coding:utf-8

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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

def print_measure_result(clf, x_test, y_test):    
    '''''测试结果的打印'''  
    answer = clf.predict(x_test)
    TP, FP, TN, FN = perf_measure(list(y_test), list(answer))
    print("TP:%d\tFP:%d\tTN:%d\tFN:%d" %(TP, FP, TN, FN))
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    specificity = TN / (TN+FP)
    print("specificity: %f" %specificity)
    print("recall: %f" %recall)
    print()

    
    #"""准确率和召回率"""
    #precision, recall, thresholds = precision_recall_curve(y_test, clf.predict(x_test))
    #answer = clf.predict_proba(data)[:,1]
    #print(classification_report(resMat, answer, target_names=['yes', 'no']))

# 去除无关列，填充缺失值（以中位数方式填充），拆分数据与结果
def fill_blank_and_split_result(data):
    from sklearn.preprocessing import Imputer

    data.pop("编号")
    data.pop("编号_A")
    resMat = data.pop("预测变量")
    imputer = Imputer(strategy='median')
    imputer.fit(data)
    X = imputer.transform(data)
    return pd.DataFrame(X), resMat

# 这个函数只在确认最佳模型时使用，确定模型后无需再运行
def get_best_tree_module(x_train, y_train):
    from sklearn.model_selection import GridSearchCV

    param = {'criterion':['gini', 'entropy'],
             'max_depth':[5, 10, 20, 30, 50],
             'min_samples_leaf':[1,2,3,5],
             'min_impurity_decrease':[0.05, 0.1, 0.2, 0.3, 0.5]}
    
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid=param, cv=6)
    grid.fit(x_train, y_train)
    print("最优分类器: ", grid.best_params_, "最优分数: ", grid.best_score_)

def print_tree_png(data, clf):
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

def main():
    excel_origin = pd.read_excel("test.xlsx")
    data, resMat = fill_blank_and_split_result(excel_origin)
    #data_train, data_test, res_train, res_test = train_test_split(data, resMat, test_size=0.2)
    for i in range(10):
        data_train, data_test, res_train, res_test = train_test_split(data, resMat, test_size=0.2)
        get_best_tree_module(data_train, res_train)


if __name__ == "__main__":
    main()
