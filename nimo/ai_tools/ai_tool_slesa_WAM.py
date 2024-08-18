import numpy as np
import random
import copy
import csv
import physbo
import itertools
import sys
import os

class SLESA_WAM():
    """Class of SLESA WAM

    This class can select the next candidates by random exploration.

    """

    def __init__(self, input_file, num_discretize = None, 
                 y_plot_range = None):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for candidates for MI algorithm
            num_objectives (int): the number of objectives

        """

        self.input_file = input_file
        self.num_objectives = 1

        self.num_discretize = num_discretize
        self.E_plot_max = float(y_plot_range[1])
        self.E_plot_min = float(y_plot_range[0])

    def load_data(self):
        """Loading candidates

        This function do not depend on robot.

        Returns:
            t_train (list[float]): the list where observed objectives are stored
            X_all (list[float]): the list where all descriptors are stored
            train_actions (list[float]): the list where observed actions are stored
            test_actions (list[float]): the list where test actions are stored

        """

        arr = np.genfromtxt(self.input_file, skip_header=1, delimiter=',')

        arr_train = arr[~np.isnan(arr[:, - 1]), :]
        arr_test = arr[np.isnan(arr[:, - 1]), :]


        X_train = arr_train[:, : - self.num_objectives]
        t_train = arr_train[:, - self.num_objectives:]

        X_test = arr_test[:, : - self.num_objectives]

        test_actions = np.where(np.isnan(arr[:, -1]))[0].tolist()

        X_all=arr[:, : - self.num_objectives]

        all_actions = [i for i in range(len(X_all))]

        train_actions = np.sort(list(set(all_actions) - set(test_actions)))

        return t_train, X_all, train_actions, test_actions



    def calculation(self):
        """Calculation of WAM

        Returns:
            True (str) for success.

        """

        print("Start analysis by WAM!")

        t_train, X_all, train_actions, test_actions = self.load_data()


        with open('slesa_log_index.csv') as f:
            reader = csv.reader(f)
            l = [row for row in reader]

        with open('slesa_log_e.csv') as f:
            reader = csv.reader(f)
            ene = [row for row in reader]


        #min-max standardization by initial data
        E_stand = []

        for ii in range(len(l)-1):
            index = list(train_actions).index(int(l[ii+1][0]))

            E_stand.append(list(itertools.chain.from_iterable(t_train))[index])


        E_min = min(E_stand)
        E_max = max(E_stand)


        allenergy = np.zeros([len(l[0]), len(l) - 1])

        allenergy[0] = np.array([(E_max - E_stand[i])/(E_max - E_min) for i in range(len(E_stand))])



        #resamling
        for jj in range(len(l[0]) - 1):

            E_real = []
            E_pred = []

            for ii in range(len(l)-1):

                index = list(train_actions).index(int(l[ii+1][jj+1]))
                E_real_each = list(itertools.chain.from_iterable(t_train))[index]
                E_real.append((E_max - E_real_each)/(E_max - E_min))

                E_pred.append(float(ene[ii+1][jj]))


            prob = np.exp(- float(l[0][jj+1]) * (np.array(E_real) - np.array(E_pred)))
            prob = prob/np.sum(prob)


            id = np.random.choice(len(E_pred), len(E_pred), p=prob, replace=True)

            allenergy[jj+1] = np.array(E_real)[id]

        
        num_discretize = self.num_discretize
        E_plot_max = self.E_plot_max
        E_plot_min = self.E_plot_min

        #make histogram
        estdists = np.zeros([len(l[0]),num_discretize])

        Eall_max = (E_max - E_plot_min)/(E_max - E_min)
        Eall_min = (E_max - E_plot_max)/(E_max - E_min)


        for biter in range(len(l[0])):
            for e in allenergy[biter]:
                index_current = int((e-Eall_min)/(Eall_max-Eall_min)*num_discretize)
                estdists[biter][index_current] += 1

        #multiple histogram
        estsum = np.sum(estdists,axis = 0)
        es = np.zeros(num_discretize)
        width = Eall_max-Eall_min

        for i in range(num_discretize):
            low = Eall_min + i*width/num_discretize
            high = Eall_min + (i+1)*width/num_discretize
            es[i] = (high+low)/2

        f = np.zeros(len(l[0]))
        for fiter in range(100):
            estn = np.zeros(num_discretize)
            for i in range(num_discretize):
                res = 0
                for j in range(len(l[0])):
                    res = res + sum(estdists[j,:])*np.exp(-float(l[0][j])*es[i]+f[j])
                estn[i] = estsum[i]/res

            for j in range(len(l[0])):
                f[j] = -np.log(np.dot(estn, np.exp(-float(l[0][j])*es)))

        estn = estn/np.sum(estn)


        real_notch_E = []

        for i in range(num_discretize):
            real_notch_E.append(- es[i]* (E_max - E_min) + E_max)

        real_notch_E.reverse()
        estn = list(estn)
        estn.reverse()

                        
        with open('res_slesa_WAM.csv', 'w', newline = "") as f:
            writer = csv.writer(f)
            writer.writerow(real_notch_E)
            writer.writerow(estn)

        print("Finish calculations!")

        return "True"
