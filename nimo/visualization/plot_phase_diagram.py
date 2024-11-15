import numpy as np
import csv

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from mpl_toolkits.mplot3d import Axes3D

import sklearn.semi_supervised
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance

def plot(input_file, fig_folder = None):
    """Loading candidates

    This function do not depend on robot.

    Args:
        input_file (str): the file for candidates for MI algorithm

    """
    
    if fig_folder == None:
        fig_path = "./fig"
    else:
        fig_path = fig_folder

    t_train, X_all, train_actions, test_actions = load_data(input_file)


    #parameters
    LP_algorithm = 'LP' #'LP', 'LS'                                                                   


    #Preparation of data    
    data_list = np.array(X_all)

    label_list = []
    for i in range(len(data_list)):
        if i in train_actions:
            phase = int(t_train[train_actions.index(i)])

        else:
            phase = - 1
        label_list.append(phase)

    label_index_list = range(len(data_list))
    labeled_index_list = train_actions
    unlabeled_index_list = test_actions
    dimension = len(data_list[0])

        
    ss = StandardScaler()
    ss.fit(data_list)
    data_list_std = ss.transform(data_list)

    max_label = np.max(list(set(label_list)))
    color_list = [cm.rainbow(float(i)/(max_label)) for i in range(max_label+1)]


    #----SAMPLING
    label_train = np.copy(label_list)

    #estimate phase of each point
    if LP_algorithm == 'LS':
        #lp_model = label_propagation.LabelSpreading()
        lp_model = sklearn.semi_supervised.LabelSpreading()
    elif LP_algorithm == 'LP':
        #lp_model = label_propagation.LabelPropagation()
        lp_model = sklearn.semi_supervised.LabelPropagation()

    lp_model.fit(data_list_std, label_train)
    #predicted_labels = lp_model.transduction_[unlabeled_index_list]
    predicted_all_labels = lp_model.transduction_
    #label_distributions = lp_model.label_distributions_[unlabeled_index_list]
    #label_distributions_all = lp_model.label_distributions_
    #classes = lp_model.classes_


    dt_now = time.localtime()

    if len(data_list[0]) == 2:
        
        fig = plt.figure()

        #for i in unlabeled_index_list:
        #    plt.scatter(data_list[i][0], data_list[i][1], c=[color_list[predicted_all_labels[i]]], marker = 's')

        for i in labeled_index_list:
            plt.scatter(data_list[i][0], data_list[i][1], c=[color_list[label_list[i]]], marker="o")

        plt.xlabel("x")
        plt.ylabel("y")
        
        plt.savefig(fig_path + "/phase_" + time.strftime('%y%m%d%H%M%S', dt_now) + ".png")
        plt.clf()
        plt.close() 
        

        fig = plt.figure()

        for i in unlabeled_index_list:
            plt.scatter(data_list[i][0], data_list[i][1], c=[color_list[predicted_all_labels[i]]], marker = 's')

        for i in labeled_index_list:
            plt.scatter(data_list[i][0], data_list[i][1], c=[color_list[label_list[i]]], marker="o")

        plt.xlabel("x")
        plt.ylabel("y")
        
        plt.savefig(fig_path + "/phase_diagram_" + time.strftime('%y%m%d%H%M%S', dt_now) + ".png")
        plt.clf()
        plt.close() 


    if len(data_list[0]) == 3:
        
        fig = plt.figure()
        ax = Axes3D(fig)

        fig.add_axes(ax)

        #for i in unlabeled_index_list:
        #    ax.scatter(data_list[i][0], data_list[i][1], data_list[i][2], c=[color_list[predicted_all_labels[i]]], marker = 's', alpha = 0.7)

        for i in labeled_index_list:
            ax.scatter(data_list[i][0], data_list[i][1], data_list[i][2], c=[color_list[label_list[i]]], marker="o", alpha = 0.7)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
     
        plt.savefig(fig_path + "/phase_" + time.strftime('%y%m%d%H%M%S', dt_now) + ".png")
        plt.clf()
        plt.close() 


        fig = plt.figure()
        ax = Axes3D(fig)
        
        fig.add_axes(ax)

        for i in unlabeled_index_list:
            ax.scatter(data_list[i][0], data_list[i][1], data_list[i][2], c=[color_list[predicted_all_labels[i]]], marker = 's', alpha = 0.7)

        for i in labeled_index_list:
            ax.scatter(data_list[i][0], data_list[i][1], data_list[i][2], c=[color_list[label_list[i]]], marker="o", alpha = 0.7)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        plt.savefig(fig_path + "/phase_diagram_" + time.strftime('%y%m%d%H%M%S', dt_now) + ".png")
        plt.clf()
        plt.close() 



def load_data(input_file):
    """Loading candidates

    This function do not depend on robot.

    Args:
        input_file (str): the file for candidates for MI algorithm

    Returns:
        t_train (list[float]): the list where observed objectives are stored
        X_all (list[float]): the list where all descriptors are stored
        train_actions (list[float]): the list where observed actions are stored
        test_actions (list[float]): the list where test actions are stored

    """

    num_objectives = 1

    arr = np.genfromtxt(input_file, skip_header=1, delimiter=',')

    arr_train = arr[~np.isnan(arr[:, - 1]), :]
    arr_test = arr[np.isnan(arr[:, - 1]), :]


    X_train = arr_train[:, : - num_objectives]
    t_train = arr_train[:, - num_objectives:]

    X_test = arr_test[:, : - num_objectives]

    test_actions = np.where(np.isnan(arr[:, -1]))[0].tolist()

    X_all=arr[:, : - num_objectives]

    all_actions = [i for i in range(len(X_all))]

    train_actions = list(np.sort(list(set(all_actions) - set(test_actions))))

    return t_train, X_all, train_actions, test_actions



