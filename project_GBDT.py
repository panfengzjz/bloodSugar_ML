#coding:utf-8
#xgboost
#from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib  #保存模型

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
    print("TP:%d\tFP:%d\tTN:%d\tFN:%d" %(TP, FP, TN, FN))
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    specificity = TN / (TN+FP)
    print("specificity: %f" %specificity)
    print("recall: %f" %recall)
    print("sum of recall & specificity is %f" %(specificity+recall))
    print()

# 去除无关列，填充缺失值（以中位数方式填充），拆分数据与结果
def fill_blank_and_split_result(data):
    from sklearn.preprocessing import Imputer

    data.pop("编号")
    data.pop("编号_A")
    data.pop("性别1男2女")
    resMat = data.pop("预测变量")
    imputer = Imputer(strategy='median')
    imputer.fit(data)
    X = imputer.transform(data)
    return pd.DataFrame(X), resMat

# 这个函数只在确认最佳模型时使用，确定模型后无需再运行
def get_best_tree_module(x_train, y_train):
    from sklearn.model_selection import GridSearchCV

    param = {'max_depth': [5],
             'min_samples_split': [3, 5, 8, 10],
             'min_samples_leaf': [3],
             'n_estimators': [100],
             #'learning_rate': [1, 2, 3, 5, 8, 10],
             'subsample': [0.3, 0.5, 0.6, 0.8, 1.0],
             'loss': ['exponential'],
             'criterion': ['friedman_mse']}
    
    grid = GridSearchCV(GradientBoostingClassifier(), param_grid=param, cv=5, n_jobs=4)
    grid.fit(x_train, y_train)
    print("最优分类器: ", grid.best_params_, "最优分数: ", grid.best_score_)
    #joblib.dump(grid, "train_module_20190824.m")

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
    gbdt = GradientBoostingClassifier(
        loss='exponential',          #GBDT算法中的损失函数
        learning_rate=0.1,           #步长，用步长和迭代最大次数一起来决定算法的拟合效果
        n_estimators=100,            #弱学习器的最大迭代次数
        subsample=0.8,               #子采样，取值为(0,1]，推荐在[0.5, 0.8]之间，默认是1.0，即不使用子采样
        criterion='friedman_mse',    #
        min_samples_split=5,         #内部节点再划分所需最小样本数，如果样本量数量级非常大，则推荐增大这个值
        min_samples_leaf=3,          #叶子节点最少样本数，如果样本量数量级非常大，则推荐增大这个值
        min_weight_fraction_leaf=0., 
        max_depth=5,                 #决策树最大深度，模型样本量多，特征也多的情况下，推荐限制这个最大深度
        max_leaf_nodes=None,         #最大叶子节点数，但是如果特征分成多的话，可以加以限制，防止过拟合
        min_impurity_decrease=0.)

    #joblib.dump(gbdt, "train_module_20190824.m")
    #gbr = joblib.load("train_module_20190824.m")
 
    for i in range(8):
        data_train, data_test, res_train, res_test = train_test_split(data, resMat, test_size=0.2)
        #get_best_tree_module(data_train, res_train)
        gbdt.fit(data_train, res_train)
        print_measure_result(gbdt, data_test, res_test)


if __name__ == "__main__":
    main()
