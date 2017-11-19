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
from sklearn.datasets import load_svmlight_files
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
    print("KNNeighbour : ", scores.mean())

    randomForest = RandomForestClassifier()
    scores = model_selection.cross_val_score(randomForest, data, target, cv=10)
    print("RForest : ", scores.mean())

    nBayes = naive_bayes.GaussianNB()
    scores = model_selection.cross_val_score(nBayes, data, target, cv=10)
    print("Naive Bayes : ", scores.mean())

    logR = linear_model.LogisticRegression()
    scores = model_selection.cross_val_score(logR, data, target, cv=10)
    print("Log R : ", scores.mean())


def runModelSelectionRandomForest(data, target):
    param_grid = [{'n_estimators': list(range(10, 400, 30)), 'criterion': ["gini", "entropy"],
                   "max_features": ["auto", "log2", "sqrt"]}]

    clf = GridSearchCV(RandomForestClassifier(random_state=10), param_grid, cv=10)

    clf.fit(data, target)

    print("\n Best parameters set found on development set:")

    print(clf.best_params_, "with a score of ", clf.best_score_)

    return clf.best_estimator_


def runModelSelectionKNN(data, target):
    param_grid = [{'n_neighbors': list(range(1, 30, 2)), 'p': [1, 2, 3, 4, 5], "weights": ["uniform", "distance"]}]

    clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)

    clf.fit(data, target)

    print("\n Best parameters set found on development set:")

    print(clf.best_params_, "with a score of ", clf.best_score_)

    return clf.best_estimator_


def runModelSelectionSVC(data, target):
    knn = SVC()
    scores = model_selection.cross_val_score(knn, data, target, cv=10)
    print(scores.mean())

    Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    gammas = [0.001, 0.01, 0.1, 1]
    kernel = ['linear', 'rbf']

    param_grid = {'C': Cs, 'gamma': gammas, 'kernel': kernel}

    clf = GridSearchCV(SVC(), param_grid, cv=10)

    clf.fit(data, target)

    print("\n Best parameters set found on SVC development set:")

    print(clf.best_params_, "with a score of ", clf.best_score_)

    return clf.best_estimator_


def performPreprocessing(gender):

    # Encode Categorical Variables by transforming the labels strings to ints
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
    gender_train = pd.read_csv("res/train_voice.csv", delimiter=",")
    feature_train = performPreprocessing(gender_train)

    gender_test = pd.read_csv("res/test_voice.csv", delimiter=",")
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

    # Remove gender label from test data and store as a Series object
    genderSeries = feature_test["label"]
    feature_test = feature_test.drop(["label"], axis=1)

    runClassifiersCV(feature_train, label_train)

    # print("FEATURE", feature_train.head())
    # print("LABEL", label_train.head())

    bestModel = runModelSelectionRandomForest(feature_train, label_train)
    # bestModel = runModelSelectionKNN(feature_train, label_train)

    # {'C': 1, 'gamma': 0.001, 'kernel': 'linear'} with a score of  0.9696969696969697
    # bestModel = runModelSelectionSVC(feature_train, label_train)


    results = bestModel.predict(feature_test)

    resultSeries = pd.Series(data=results, name='label', dtype='int64')

    df = pd.DataFrame({"PassengerId": genderSeries, "label": resultSeries})
    df.to_csv("res/kaggle_gridSearch.csv", index=False, header=True)

if __name__ == '__main__':
    main()