#!/usr/bin/env python
import argparse
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import knn as knn


from sklearn.tree import DecisionTreeClassifier

# make sure we're working in the directory this file lives in,
# for imports and for simplicity with relative paths
os.chdir(Path(__file__).parent.resolve())

# our code
from utils import load_dataset, plot_classifier, handle, run, main
from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from kmeans import Kmeans
from knn import KNN
from naive_bayes import NaiveBayes, NaiveBayesLaplace
from random_tree import RandomForest, RandomTree


@handle("1")
def q1():
    dataset = load_dataset("citiesSmall.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]

    """YOUR CODE HERE FOR Q1. Also modify knn.py to implement KNN predict."""
    k1 = KNN(1)
    k1.fit(X, y)
    y_predTrain1 = k1.predict(X)
    errort1 = np.mean(y_predTrain1 != y)
    print("training error k = 1:", errort1)
    k3 = KNN(3)
    k3.fit(X, y)
    y_predTrain1 = k3.predict(X)
    errort3 = np.mean(y_predTrain1 != y)
    print("training error k = 3:", errort3)

    k10 = KNN(10)
    k10.fit(X, y)
    y_predTrain1 = k10.predict(X)
    errort10 = np.mean(y_predTrain1 != y)
    print("training error k = 10:", errort10)
    k1 = KNN(1)
    k1.fit(X, y)
    y_predTrain1 = k1.predict(X_test)
    errort1 = np.mean(y_predTrain1 != y_test)
    print("testing error k = 1:", errort1)
    k3 = KNN(3)
    k3.fit(X, y)
    y_predTrain1 = k3.predict(X_test)
    errort3 = np.mean(y_predTrain1 != y_test)
    print("testing error k = 3:", errort3)

    k10 = KNN(10)
    k10.fit(X, y)
    y_predTrain1 = k10.predict(X_test)
    errort10 = np.mean(y_predTrain1 != y_test)
    print("testing error k = 10:", errort10)

    plot_classifier(k1, X, y)

    fname = Path("..", "figs", "Q1.3.pdf")
    plt.savefig(fname)



@handle("2")
def q2():
    dataset = load_dataset("ccdebt.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    cv_accs = np.zeros(8)
    ks = list(range(1, 30, 4))
    size = len(X) // 10
    n,d = X.shape
    test_error = np.zeros(8)
    """YOUR CODE HERE FOR Q2"""
    for i in ks:

        k = KNN(i)
        m = np.zeros(10)
        k.fit(X, y)
        test_error[ks.index(i)] = np.mean(y_test != k.predict(X_test))
        for j in range(1,11):
            mask = np.ones(n, dtype = bool)
            mask[size*(j-1): size*j] = False
            trainX = X[mask]
            trainY = y[mask]
            testX = X[~mask]
            testY = y[~mask]
            k.fit(trainX, trainY)
            y_pred1 = k.predict(testX)
            m[j-1] = np.mean(y_pred1 != testY)
        cv_accs[ks.index(i)] = np.mean(m)

    print(cv_accs)
    print(test_error)
    plt.plot(ks, cv_accs, label="cross-validation")
    plt.xlabel("K # of neighbours")
    plt.ylabel("cross-validation error")
    plt.legend()
    fname = Path("..", "figs", "q2.2.pdf")
    plt.savefig(fname)

    plt.plot(ks, test_error, label = "testing")
    plt.xlabel("K # of neighbours")
    plt.ylabel("test error")
    plt.legend()
    fname = Path("..", "figs", "q2.3.pdf")
    plt.savefig(fname)
@handle("3.2")
def q3_2():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"].astype(bool)
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]
    groupnames = dataset["groupnames"]
    wordlist = dataset["wordlist"]

    """YOUR CODE HERE FOR Q3.2"""

    print(groupnames[y[802]])
    print(groupnames)



@handle("3.3")
def q3_3():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    """CODE FOR Q3.4: Modify naive_bayes.py/NaiveBayesLaplace"""

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)

    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")


@handle("3.4")
def q3_4():
    dataset = load_dataset("newsgroups.pkl")

    X = dataset["X"]
    y = dataset["y"]
    X_valid = dataset["Xvalidate"]
    y_valid = dataset["yvalidate"]

    print(f"d = {X.shape[1]}")
    print(f"n = {X.shape[0]}")
    print(f"t = {X_valid.shape[0]}")
    print(f"Num classes = {len(np.unique(y))}")

    model = NaiveBayes(num_classes=4)
    model.fit(X, y)
    y_hat = model.predict(X)
    err_train = np.mean(y_hat != y)
    print(f"Naive Bayes training error: {err_train:.3f}")

    y_hat = model.predict(X_valid)
    err_valid = np.mean(y_hat != y_valid)
    print(f"Naive Bayes validation error: {err_valid:.3f}")

    model1 = NaiveBayesLaplace(4, 1)
    model1.fit(X, y)
    y_hat1 = model1.predict(X)
    err_train1 = np.mean(y_hat1 != y)
    print(f"Naive Bayes lapace training error: {err_train1:.3f}")

    y_hat1 = model1.predict(X_valid)
    err_valid1 = np.mean(y_hat1 != y_valid)
    print(f"Naive Bayes lapace validation error: {err_valid1:.3f}")

    model2 = NaiveBayesLaplace(4, 10000)
    model2.fit(X, y)
    y_hat2 = model2.predict(X)
    err_train2 = np.mean(y_hat1 != y)
    print(f"Naive Bayes lapace (10000) training error: {err_train2:.3f}")

    y_hat2 = model2.predict(X_valid)
    err_valid2 = np.mean(y_hat2 != y_valid)
    print(f"Naive Bayes lapace (10000) validation error: {err_valid2:.3f}")
    """YOUR CODE HERE FOR Q3.4. Also modify naive_bayes.py/NaiveBayesLaplace"""




@handle("4")
def q4():
    dataset = load_dataset("vowel.pkl")
    X = dataset["X"]
    y = dataset["y"]
    X_test = dataset["Xtest"]
    y_test = dataset["ytest"]
    print(f"n = {X.shape[0]}, d = {X.shape[1]}")

    def evaluate_model(model):
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print(f"    Training error: {tr_error:.3f}")
        print(f"    Testing error: {te_error:.3f}")

    print("Decision tree info gain")
    evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
    print("Random tree")
    evaluate_model(RandomTree(np.inf))
    print("Random Forest")
    evaluate_model(RandomForest(50, np.inf))
    """YOUR CODE FOR Q4. Also modify random_tree.py/RandomForest"""




@handle("5")
def q5():
    X = load_dataset("clusterData.pkl")["X"]

    model = Kmeans(k=4)
    model.fit(X)
    y = model.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

    fname = Path("..", "figs", "kmeans_basic_rerun.png")
    plt.savefig(fname)
    print(f"Figure saved as {fname}")


@handle("5.1")
def q5_1():
    X = load_dataset("clusterData.pkl")["X"]
    model = Kmeans(k=4)
    """YOUR CODE HERE FOR Q5.1. Also modify kmeans.py/Kmeans"""
    # for i in range(4):

    model.fit(X)
    # print("error:", model.error(X,y,model.means))
    y = model.predict(X)
    # print("error:", model.error(X,y,model.means))


@handle("5.2")
def q5_2():
    X = load_dataset("clusterData.pkl")["X"]
    allMinError = np.zeros(10)
    ks = [1,2,3,4,5,6,7,8,9,10]
    for i in ks:
        models = [Kmeans(k=i) for _ in range(50)]
        for j in range(50):
            models[j].fit(X)
        allY = np.array([model.predict(X) for model in models])
        allerrors = np.ones(50)
        for j in range(50):
            allerrors[j] = models[j].error(X, allY[j], models[j].means)
        allMinError[i-1] = np.min(allerrors)
        print(allMinError[i-1])



    """YOUR CODE HERE FOR Q5.2"""

    plt.plot(ks, allMinError, label="Min errors")
    plt.xlabel("K # of clusters")
    plt.ylabel("min error")
    plt.legend()
    fname = Path("..", "figs", "q5.2.pdf")
    plt.savefig(fname)



if __name__ == "__main__":
    main()
