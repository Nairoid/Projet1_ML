"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classical algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier

from data import make_data1, make_data2
from plot import plot_boundary
from matplotlib import pyplot as plt


if __name__ == "__main__":

    accuracies1 = []
    accuracies2 = []
    for k in range(0, 5, 1):  # five generations of dataset
        X_train1, y_train1, X_test1, y_test1 = make_data1(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=k)
        X_train2, y_train2, X_test2, y_test2 = make_data2(n_ts=10000, n_ls=250, noise=0, plot=False, random_state=k)

        acc1 = []
        acc2 = []

        for i in range(0, 4, 1):  # different max_depths
            j = pow(2, i)
            filename1 = "dt_data1_maxdepth" + str(j)
            plt_title1 = "Dataset 1: max_depth = " + str(j)
            filename2 = "dt_data2_maxdepth" + str(j)
            plt_title2 = "Dataset 2: max_depth = " + str(j)
            dtc1 = DecisionTreeClassifier(max_depth=j).fit(X_train1, y_train1)
            plot_boundary(filename1, dtc1, X_train1, y_train1, mesh_step_size=0.1, title=plt_title1,
                          inline_plotting=False)
            acc1.append(dtc1.score(X_test1, y_test1))
            dtc2 = DecisionTreeClassifier(max_depth=j).fit(X_train2, y_train2)
            plot_boundary(filename2, dtc2, X_train2, y_train2, mesh_step_size=0.1, title=plt_title2,
                          inline_plotting=False)
            acc2.append(dtc2.score(X_test2, y_test2))
        filename1 = "dt_data1_maxdepth_None"
        plt_title1 = "Dataset 1: max_depth = None"
        dtc1 = DecisionTreeClassifier(max_depth=None).fit(X_train1, y_train1)
        plot_boundary(filename1, dtc1, X_train1, y_train1, mesh_step_size=0.1, title=plt_title1,
                      inline_plotting=False)
        acc1.append(dtc1.score(X_test1, y_test1))

        filename2 = "dt_data2_maxdepth_None"
        plt_title2 = "Dataset 2: max_depth = None"
        dtc2 = DecisionTreeClassifier(max_depth=None).fit(X_train2, y_train2)
        plot_boundary(filename2, dtc2, X_train2, y_train2, mesh_step_size=0.1, title=plt_title2,
                      inline_plotting=False)
        acc2.append(dtc2.score(X_test2, y_test2))

        accuracies1.append(np.array(acc1))
        accuracies2.append(np.array(acc2))

    accuracies1 = np.array(accuracies1)
    means1 = np.mean(accuracies1, axis=0)
    stds1 = np.std(accuracies1, axis=0)

    accuracies2 = np.array(accuracies2)
    means2 = np.mean(accuracies2, axis=0)
    stds2 = np.std(accuracies2, axis=0)

    lstt = ["1", "2", "4", "8", "None"]
    plt.errorbar(lstt, means1, stds1, linestyle=':', marker='.', color='cornflowerblue', ecolor='mediumblue',
                 label='Dataset 1')
    plt.errorbar(lstt, means2, stds2, linestyle=':', marker='.', color='darkred', ecolor='tomato', label='Dataset 2')
    plt.legend()
    plt.title("Average test set accuracies")
    plt.xlabel("Complexity of the tree (max_depth)")
    plt.ylabel("Average test set accuracies")
    plt.savefig("avg_ex_1.png")

    df = pd.DataFrame(np.transpose([means1, means2, stds1, stds2]), columns=["Means1", "Means2", "Std1", "Std2"])
    #df.to_csv("avg_values_ex1.csv", index=False)
    print(df.to_latex(index=False))
