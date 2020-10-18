"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scipy.optimize import minimize

from sklearn.neighbors import KNeighborsClassifier

from data import make_data1, make_data2
from plot import plot_boundary

from matplotlib import pyplot as plt


# (Question 2)

def divide(data, k, i):
    """
    Divides the dataset into k parts,
    output = two arrays, one is the i^th part the other one is the given dataset w/o i^th part

    :param k:   int < n
                number of divisions
    :param data: array of length n
    :param i:   int < k
                i^th part
    :return:

    l_sample:     array of length n/k
                i^th part of the dataset
    t_sample:     array of length n-(n/k)
                remaining data as one dataset
    """

    new_data = np.array_split(data, k)
    l_sample = new_data[i]
    t_sample = []
    t_sample_hold = np.delete(new_data, i, axis=0)
    for elements in t_sample_hold:
        t_sample.extend(elements.tolist())

    np.asarray(t_sample)
    return l_sample, t_sample


def kcv_score(mc, X_data, y_data, k=5):
    """
    Gives the mean k-fold cross validation score of dataset
    for a given model classifier

    :param X_data:  array of length n
                    input data

    :param y_data:  array of length n
                    output data

    :param mc:      class
                    model classifier

    :param k:       int < n
                    k-fold cross validation

    :return:

    mean_score:          float
                    mean score value

    """
    score = 0.
    for i in range(0, k, 1):
        X_test, X_train = divide(X_data, k, i)
        y_test, y_train = divide(y_data, k, i)
        mc2 = mc.fit(X_train, y_train)
        score += mc2.score(X_test, y_test)
    mean_score = score / k
    return mean_score


def find_min(X_data, y_data, lb=1, ub=50):
    """
    Finds the optimal number of neighbors in range [lb, ub)
    for given dataset

    :param X_data:  array of length [n,2]
    :param y_data:  array of length n
    :param lb:      int
                    lower bound for max search
    :param ub:      int > lb
                    upper bound for max search
    :return:

    scores:         array of length ub - lb
                    Contains the kcc scores for each number of neighbors
    opt_n_n:        Int
                    optimal number of neighbors
    best_score:     float
                    value of the k-fold cross validation accuracy for the otpimal number of neighbors
    """
    scores = []
    for i in range(lb, ub, 1):
        knc = KNeighborsClassifier(n_neighbors=i)
        scores.append(kcv_score(knc, X_data, y_data))

    opt_n_n = scores.index(max(scores)) + lb
    best_score = max(scores)

    return scores, opt_n_n, best_score


if __name__ == "__main__":
    ###
    #  2.1
    ###
    X_train1, y_train1, X_test1, y_test1 = make_data1(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0)
    X_train2, y_train2, X_test2, y_test2 = make_data2(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=0)

    for i in [1, 5, 10, 75, 100, 150]:
        knc = KNeighborsClassifier(n_neighbors=i)

        knc1 = knc.fit(X_train1, y_train1)
        filename1 = "knn_data1_" + str(i)
        plt_title1 = "Dataset 1: n_neighbors = " + str(i)
        plot_boundary(filename1, knc1, X_train1, y_train1, mesh_step_size=0.1, title=plt_title1,
                      inline_plotting=False)

        # print(knc1.score(X_test1, y_test1))

        knc2 = knc.fit(X_train2, y_train2)
        filename2 = "knn_data2_" + str(i)
        plt_title2 = "Dataset 2: n_neighbors = " + str(i)
        plot_boundary(filename2, knc2, X_train2, y_train2, mesh_step_size=0.1, title=plt_title2,
                      inline_plotting=False)

    ###
    #  2.2
    ###

    y_data = np.concatenate((y_train2, y_test2))
    X_data = np.concatenate((X_train2, X_test2))
    m_kcv, n_n_opt, kcv_max = find_min(X_data, y_data, 1, 250)

    plt.plot(m_kcv)
    plt.plot(n_n_opt - 1, kcv_max, '.r', label="n="+str(n_n_opt - 1)+" accuracy="+str(kcv_max))
    plt.title("Five-fold cross validation accuracy for varying number of neighbors")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.legend(loc=8)
    plt.savefig("2_2_scores.png")
    plt.show()

    ###
    #  2.3
    ###

    acc1_best = []
    acc2_best = []
    for i in [50, 200, 250, 500]:
        X_train1, y_train1, X_test1, y_test1 = make_data1(n_ts=500, n_ls=i, noise=0.2, plot=False, random_state=0)
        X_train2, y_train2, X_test2, y_test2 = make_data2(n_ts=500, n_ls=i, noise=0.2, plot=False, random_state=0)

        acc1 = []
        acc2 = []

        for j in range(1, i+1, 1):
            knc = KNeighborsClassifier(n_neighbors=j)

            knc1 = knc.fit(X_train1, y_train1)
            acc1.append(knc1.score(X_test1, y_test1))

            knc2 = knc.fit(X_train2, y_train2)
            acc2.append(knc2.score(X_test2, y_test2))

        best1 = max(acc1)
        best2 = max(acc2)
        acc1_best.append(acc1.index(best1) + 1)
        acc2_best.append(acc2.index(best2) + 1)

        plt.plot(acc1, label="Dataset 1")
        plt.plot(acc2, label="Dataset 2")
        plt.legend()
        plt.title("Learning sample = " + str(i) + " Test sample = 500")
        plt.xlabel("Number of neighbors")
        plt.ylabel("Accuracy")
        plt.savefig("knn_scores_ls_" + str(i) + ".png")
        plt.show()

    x = ["50", "200", "250", "500"]
    plt.plot(x, acc1_best, marker='.', label="Dataset 1")
    plt.plot(x, acc2_best, marker='.', label="Dataset 2")
    plt.legend()
    plt.title("Optimal value of n_neighbors")
    plt.xlabel("Learning sample size")
    plt.ylabel("Number of neigbors")
    plt.savefig("knn_best_nn.png")
    plt.show()





