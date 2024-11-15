import numpy as np
import csv

import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D


def plot(input_file, num_objectives, fig_folder = None):
    """Loading candidates

    This function do not depend on robot.

    Args:
        input_file (str): the file for candidates for MI algorithm
        num_objectives (int): the number of objectives

    """

    if fig_folder == None:
        fig_path = "./fig"
    else:
        fig_path = fig_folder

    t_train, X_all, train_actions, test_actions = load_data(input_file, num_objectives)

    dt_now = time.localtime()

    if num_objectives == 1:
        
        fig = plt.figure()

        plt.hist(t_train)
        plt.xlabel("Objectives")
        plt.ylabel("Counts")
        plt.savefig(fig_path + "/distribution_" + time.strftime('%y%m%d%H%M%S', dt_now) + ".png")
        plt.clf()
        plt.close() 


    if num_objectives == 2:
        
        x = []
        y = []

        for i in range(len(t_train)):
            x.append(t_train[i][0])
            y.append(t_train[i][1])

        fig = plt.figure(figsize = [4.8, 4.8])

        plt.scatter(x, y)
        plt.xlabel("Objective 1")
        plt.ylabel("Objective 2")
        plt.savefig(fig_path + "/distribution_" + time.strftime('%y%m%d%H%M%S', dt_now) + ".png")
        plt.clf()
        plt.close() 


    if num_objectives == 3:
        
        x = []
        y = []
        z = []

        for i in range(len(t_train)):
            x.append(t_train[i][0])
            y.append(t_train[i][1])
            z.append(t_train[i][2])

        fig = plt.figure()
        ax = Axes3D(fig)

        ax.scatter(x, y, z)
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.set_zlabel("Objective 3")
        plt.savefig(fig_path + "/distribution_" + time.strftime('%y%m%d%H%M%S', dt_now) + ".png")
        plt.clf()
        plt.close() 


def load_data(input_file, num_objectives):
    """Loading candidates

    This function do not depend on robot.

    Args:
        input_file (str): the file for candidates for MI algorithm
        num_objectives (int): the number of objectives

    Returns:
        t_train (list[float]): the list where observed objectives are stored
        X_all (list[float]): the list where all descriptors are stored
        train_actions (list[float]): the list where observed actions are stored
        test_actions (list[float]): the list where test actions are stored

    """

    arr = np.genfromtxt(input_file, skip_header=1, delimiter=',')

    arr_train = arr[~np.isnan(arr[:, - 1]), :]
    arr_test = arr[np.isnan(arr[:, - 1]), :]


    X_train = arr_train[:, : - num_objectives]
    t_train = arr_train[:, - num_objectives:]

    X_test = arr_test[:, : - num_objectives]

    test_actions = np.where(np.isnan(arr[:, -1]))[0].tolist()

    X_all=arr[:, : - num_objectives]

    all_actions = [i for i in range(len(X_all))]

    train_actions = np.sort(list(set(all_actions) - set(test_actions)))

    return t_train, X_all, train_actions, test_actions



