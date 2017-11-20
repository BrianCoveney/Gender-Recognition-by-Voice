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
import sklearn.metrics as metrics
from sklearn.datasets import load_svmlight_files
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def runClassifiersCV(data, target):
    print("\nThe following are the initial accuracies using CV 10")
    dTree = tree.DecisionTreeClassifier()
    scores = model_selection.cross_val_score(dTree, data, target, cv=10)
    print("Tree        : ", scores.mean())

    rbfSvm = SVC()
    scores = model_selection.cross_val_score(rbfSvm, data, target, cv=10)
    print("SVM         : ", scores.mean())

    nearestN = KNeighborsClassifier()
    scores = model_selection.cross_val_score(nearestN, data, target, cv=10)
    print("KNNeighbour : ", scores.mean())

    randomForest = RandomForestClassifier()
    scores = model_selection.cross_val_score(randomForest, data, target, cv=10)
    print("RForest     : ", scores.mean())

    nBayes = naive_bayes.GaussianNB()
    scores = model_selection.cross_val_score(nBayes, data, target, cv=10)
    print("Naive Bayes : ", scores.mean())

    logR = linear_model.LogisticRegression()
    scores = model_selection.cross_val_score(logR, data, target, cv=10)
    print("Log R       : ", scores.mean())


def runModelSelectionRandomForest(data, target):
    param_grid = [{'n_estimators': list(range(10, 400, 30)), 'criterion': ["gini", "entropy"],
                   "max_features": ["auto", "log2", "sqrt"]}]

    clf = GridSearchCV(RandomForestClassifier(n_jobs=2, random_state=0), param_grid, cv=10)

    clf.fit(data, target)

    print("\n Best parameters set found on RandomForest development set:")

    print(clf.best_params_, "with a score of ", clf.best_score_)

    return clf.best_estimator_


def runModelSelectionKNN(data, target):
    param_grid = [{'n_neighbors': list(range(1, 30, 2)), 'p': [1, 2, 3, 4, 5], "weights": ["uniform", "distance"]}]

    clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=10)

    clf.fit(data, target)

    print("\n Best parameters set found o KNN development set:")

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
            "maxfun","meandom","mindom","maxdom","dfrange","modindx"]] = scalingObj.fit_transform(
        gender[["meanfreq","sd","median","Q25","Q75","IQR","skew","kurt","sp.ent","sfm","mode","centroid","meanfun","minfun",
                "maxfun","meandom","mindom","maxdom","dfrange","modindx"]])

    return gender


#
# Data Cleaning
# Ref. https://en.wikipedia.org/wiki/Voice_frequency
# The voiced speech of a typical adult:
# male fundamental frequency from 85 to 180 Hz
# female fundamental frequency from 165 to 255 Hz
#
def dataCleansing(df_train, df_test):

    print("Scatter plot pre data cleansing \n"
          "Please close window to continue")
    sns.FacetGrid(df_test, hue="label", size=5) \
        .map(plt.scatter, "meanfun", "meanfreq") \
        .add_legend()
    plt.show()

    # Filtering ouliers from male category
    male_funFreq_outlier_index = df_train[((df_train['meanfun'] < 0.085) | (df_train['meanfun'] > 0.180)) &
                                              (df_train['label'] == 'male')].index
    male_funFreq_outlier_index = list(male_funFreq_outlier_index)
    df_train[((df_train['meanfun'] < 0.085) | (df_train['meanfun'] > 0.180)) & (
        df_train['label'] == 'male')].shape

    # Filtering ouliers from female category
    female_funFreq_outlier_index = df_train[
        ((df_train['meanfun'] < 0.165) | (df_train['meanfun'] > 0.255)) &
        (df_train['label'] == 'female')].index
    female_funFreq_outlier_index = list(female_funFreq_outlier_index)
    df_train[((df_train['meanfun'] < 0.165) | (df_train['meanfun'] > 0.255)) & (
        df_train['label'] == 'female')].shape

    index_to_remove = male_funFreq_outlier_index + female_funFreq_outlier_index
    print("Index to remove:",len(index_to_remove))  # prints 710

    print("Data size", df_train.shape)  # prints (3168, 21)

    df_train = df_train.drop(index_to_remove, axis=0)
    print("Data size", df_train.shape)  # prints (2458, 21)

    df_test = df_test.drop(index_to_remove, axis=0)
    print("Data size", df_test.shape)  # prints (2458, 21)

    print("Scatter plot post data cleansing \n"
          "Please close window to continue")
    sns.FacetGrid(df_test, hue="label", size=5) \
        .map(plt.scatter, "meanfun", "meanfreq") \
        .add_legend()
    plt.show()


def main():
    gender_train = pd.read_csv("res/train_voice.csv", delimiter=",")
    gender_test = pd.read_csv("res/test_voice.csv", delimiter=",")

    # There are 3168 labels split evenly between male and female. With no null values.
    print("The following is a look at the dataset:")
    print("Number of labels: ", gender_train.shape[0])
    print("Number of males:  ", gender_train[gender_train["label"] == "male"].shape[0])
    print("Number of females:", gender_train[gender_train["label"] == "female"].shape[0])
    if gender_train.isnull() is True:
        print('There are missing values')
    else:
        print('No missing values')

    dataCleansing(gender_train, gender_test)


    feature_train = performPreprocessing(gender_train)
    feature_test = performPreprocessing(gender_test)



    # Split the training dataset into features and classes. A 1D class array and 2D features array.

    label_train = feature_train["label"]
    feature_train = feature_train.drop(["label"], axis=1)

    # Remove gender label from test data and store as a Series object
    genderSeries = feature_test["label"]
    feature_test = feature_test.drop(["label"], axis=1)

    runClassifiersCV(feature_train, label_train)



    # Best parameters set found on RandomForest development set:
    # {'criterion': 'gini', 'max_features': 'auto', 'n_estimators': 70} with a score of  0.9681186868686869
    # bestModel = runModelSelectionRandomForest(feature_train, label_train)

    # Best parameters set found o KNN development set:
    # {'n_neighbors': 3, 'p': 2, 'weights': 'uniform'} with a score of  0.95864898989899
    # bestModel = runModelSelectionKNN(feature_train, label_train)

    # Best parameters set found on SVC development set:
    # {'C': 1, 'gamma': 0.001, 'kernel': 'linear'} with a score of  0.9696969696969697
    # bestModel = runModelSelectionSVC(feature_train, label_train)
    #
    # results = bestModel.predict(feature_test)
    #
    # resultSeries = pd.Series(data=results, name='label', dtype='int64')
    #
    # df = pd.DataFrame({"PassengerId": genderSeries, "label": resultSeries})
    # df.to_csv("res/kaggle_gridSearch.csv", index=False, header=True)

if __name__ == '__main__':
    main()