#coding:utf-8

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import pandas as pd

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

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
        if list_predict[i]==0 and y_actual[i]==1:
            FN += 1

    return (TP, FP, TN, FN)

def print_measure_result(tree, x_test, y_test):    
    '''''测试结果的打印'''
    answer = tree.predict(x_test)
    round_answer = []
    for i in answer:
        if (i>0.5):
            round_answer.append(1)
        else:
            round_answer.append(0)
    TP, FP, TN, FN = perf_measure(list(y_test), list(round_answer))
    #print("TP:%d\tFP:%d\tTN:%d\tFN:%d" %(TP, FP, TN, FN))
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    specificity = TN / (TN+FP)
    #print("specificity: %f" %specificity)
    #print("recall: %f" %recall)
    #print("sum of recall & specificity is %f" %(specificity+recall))
    print(specificity+recall)
    #print()

# 去除无关列，填充缺失值（以中位数方式填充），拆分数据与结果
def fill_blank_and_split_result(data):
    from sklearn.preprocessing import Imputer

    resMat = data.pop("预测变量")
    data.drop(['编号', '编号_A', '性别1男2女'], axis=1, inplace=True)
    imputer = Imputer(strategy='median')
    imputer.fit(data)
    X = imputer.transform(data)
    return pd.DataFrame(X), resMat

# 这个函数只在确认最佳模型时使用，确定模型后无需再运行
def get_best_tree_module(x_train, y_train):
    from sklearn.model_selection import GridSearchCV

    param = {'max_depth': [5],
             'min_samples_split': [6],
             'min_samples_leaf': [4],
             'n_estimators': [80,100,120,150,180,200],
             #'learning_rate': [1, 2, 3, 5, 8, 10],
             #'subsample': [0.3, 0.5, 0.6, 0.8, 1.0],
             'criterion': ['gini']
             }
    
    grid = GridSearchCV(RandomForestClassifier(), param_grid=param, cv=5)
    grid.fit(x_train, y_train)
    print("最优分类器: ", grid.best_params_, "最优分数: ", grid.best_score_)
    #joblib.dump(grid, "train_module_20190824.m")

def main():
    excel_origin = pd.read_excel("test.xlsx")
    data, resMat = fill_blank_and_split_result(excel_origin)
    rfc = RandomForestClassifier(
        n_estimators=80, 
        criterion="gini", 
        max_depth=5, 
        min_samples_split=5, 
        min_samples_leaf=5, 
        min_weight_fraction_leaf=0., 
        max_features="auto", 
        max_leaf_nodes=None, 
        min_impurity_decrease=0., 
        min_impurity_split=None
        )

    for i in range(20):
        data_train, data_test, res_train, res_test = train_test_split(data, resMat, test_size=0.2)
        #get_best_tree_module(data_train, res_train)
        rfc.fit(data_train, res_train)
        print_measure_result(rfc, data_test, res_test)

if __name__ == "__main__":
    main()
