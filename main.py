from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import naive_bayes
from sklearn.grid_search import GridSearchCV
from sklearn import model_selection
from sklearn import linear_model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def runClassifiersCV(data, target):
    print("\nThe following are the initial accuracies using CV 10")
    dTree = tree.DecisionTreeClassifier()
    scores = model_selection.cross_val_score(dTree, data, target, cv=10)
    print("Tree : ", scores.mean())

    rbfSvm = SVC()
    scores = model_selection.cross_val_score(rbfSvm, data, target, cv=10)
    print("SVM : ", scores.mean())

    nearestN = KNeighborsClassifier()
    scores = model_selection.cross_val_score(nearestN, data, target, cv=10)
    print("NNeighbour : ", scores.mean())

    randomForest = RandomForestClassifier()
    scores = model_selection.cross_val_score(randomForest, data, target, cv=10)
    print("RForest : ", scores.mean())

    nBayes = naive_bayes.GaussianNB()
    scores = model_selection.cross_val_score(nBayes, data, target, cv=10)
    print("Naive Bayes : ", scores.mean())

    logR = linear_model.LogisticRegression()
    scores = model_selection.cross_val_score(logR, data, target, cv=10)
    print("Log R : ", scores.mean())


def performPreprocessing(gender):

    # Encode Categorical Variables by transforming the labels strings to ints
    # 'male': 1, 'female': 0
    # gender = pd.get_dummies(gender, columns=["label"])
    gender["label"] = gender["label"].map({'male': 0, 'female': 1}).astype(int)

    # Normalize the Data to standardize the features
    scalingObj = preprocessing.MinMaxScaler()
    gender[["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun",
            "maxfun","meandom","mindom","maxdom","dfrange","modindx"]] = scalingObj.fit_transform(gender[["meanfreq","sd",
            "median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun","maxfun","meandom",
            "mindom","maxdom","dfrange","modindx"]])

    return gender


def main():
    gender_train = pd.read_csv("res/train_voice.csv", delimiter=",", encoding='utf8')
    feature_train = performPreprocessing(gender_train)

    gender_test = pd.read_csv("res/test_voice.csv", delimiter=",", encoding='utf8')
    feature_test = performPreprocessing(gender_test)

    # There are 3168 labels split evenly between male and female. With no null values.
    # print("Number of labels:", gender_train.shape[0])
    # print("Number of males:", gender_train[gender_train["label"] == "male"].shape[0])
    # print("Number of females:", gender_train[gender_train["label"] == "female"].shape[0])
    # print(gender_train.isnull().sum())
    # print(gender_train.head())

    # Split the training dataset into features and classes. A 1D class array and 2D features array.

    label_train = feature_train["label"]
    feature_train = feature_train.drop(["label"], axis=1)

    runClassifiersCV(feature_train, label_train)


if __name__ == '__main__':
    main()