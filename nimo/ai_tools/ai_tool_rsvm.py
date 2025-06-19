import numpy as np
import random
import copy
import csv
import itertools

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


class RSVM():
    """Class of RSVM

    This class can select the next candidates by rank SVM.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals, other_datasets, minimization, output_res):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for candidates for MI algorithm
            output_file (str): the file for proposals from MI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
            output_res (str): True or False to export prediction results

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals
        
        self.other_datasets = other_datasets
        self.minimization = minimization
        self.output_res = output_res
        
        if self.minimization == None:
            self.minimization = False

        


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



    def calc_ai(self, t_train, X_all, train_actions, test_actions):
        """Calculating the proposals by AI algorithm

        This function is for RSVM.
        This function do not depend on robot.
        If the new AI alborithm is developed, this function is only changed.
        
        Args:
            t_train (list[float]): the list where observed objectives are stored
            X_all (list[float]): the list where all descriptors are stored
            train_actions (list[float]): the list where observed actions are stored
            test_actions (list[float]): the list where test actions are stored

        Returns:
            actions (list[int]): the list where the selected actions are stored

        """

        #create training data
        sc = StandardScaler()
        sc.fit(X_all)

        X_train = []
        y_train = []


        #candidates dataset
        if len(train_actions) >= 2:
        
            features_observed = sc.transform(X_all[train_actions])
            properties_observed = t_train

            combi = list(itertools.permutations(range(len(properties_observed)), 2))


            for ii in range(len(combi)):
                X_train.append(features_observed[combi[ii][0]] - features_observed[combi[ii][1]])

                if properties_observed[combi[ii][0]] >= properties_observed[combi[ii][1]]:
                    y_train.append(1)
                else:
                    y_train.append(-1)


        #other datasets
        for jj in range(len(self.other_datasets)):
        
            arr = np.genfromtxt(self.other_datasets[jj], skip_header=1, delimiter=',')

            X_data = arr[:, : - self.num_objectives]
            t_data = arr[:, - self.num_objectives:]

            features_observed = sc.transform(X_data)
            properties_observed = t_data

            combi = list(itertools.permutations(range(len(properties_observed)), 2))

            for ii in range(len(combi)):
                X_train.append(features_observed[combi[ii][0]] - features_observed[combi[ii][1]])

                if properties_observed[combi[ii][0]] >= properties_observed[combi[ii][1]]:
                    y_train.append(1)
                else:
                    y_train.append(-1)


        params = {'C':[0.0001, 0.001, 0.01, 0.02, 0.05, 0.06, 0.1, 1,2,5, 10, 100, 1000]}
        gridsearch = GridSearchCV(LinearSVC(dual = False, max_iter = 10000), param_grid = params, cv = 5, scoring = 'accuracy', n_jobs = 1, verbose = 1)
        gridsearch.fit(X_train,y_train)

        print(f"Best score of SVC (accuracy): {gridsearch.best_score_:.3f}")

        model =  LinearSVC(C = gridsearch.best_params_['C'], dual = False, max_iter = 10000)
        model.fit(X_train, y_train)


        #Predict properties of unchecked data

        features_unchecked = sc.transform(X_all[test_actions])
        features_unchecked_prev = X_all[test_actions]
        
        predicted_properties_list = []
        for ii in range(len(features_unchecked)):
            predicted_properties_list.append(np.dot(features_unchecked[ii], model.coef_[0]))


        #Output prediction results
        if self.output_res == True:
        
            res_tot = []

            f = open(self.input_file, 'r')
            reader = csv.reader(f)
            header = next(reader)

            header = header[0 : - self.num_objectives]

            header.append('acquisition')

            res_tot.append(header)

            for ii in range(len(features_unchecked)):

                res_each = []

                for jj in range(len(features_unchecked[0])):
                    res_each.append(features_unchecked_prev[ii][jj])

                res_each.append(predicted_properties_list[ii])

                res_tot.append(res_each)


            with open('output_res.csv', 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(res_tot)


            #Select next candidates
            if self.minimization == True:
                rank_index = np.array(predicted_properties_list).argsort()[::]

            else:
                rank_index = np.array(predicted_properties_list).argsort()[::-1]


            actions = []

            for i in range(self.num_proposals):

                actions.append(test_actions[rank_index[i]])

        return actions



    def select(self):
        """Selecting the proposals by MI algorithm

        This function do not depend on robot.

        Returns:
            True (str) for success.

        """

        print("Start selection of proposals by RSVM!")

        t_train, X_all, train_actions, test_actions = self.load_data()

        actions = self.calc_ai(t_train = t_train, X_all = X_all, 
        train_actions = train_actions, test_actions = test_actions)


        print('Proposals')

        proposals_all = []

        input_data = open(self.input_file, 'r')
        indexes = input_data.readlines()[0].rstrip('\n').split(',')

        indexes = ["actions"] + indexes[0 : - self.num_objectives]

        proposals_all.append(indexes)

        for i in range(len(actions)):

            row = [str(X_all[actions[i]][j]) for j in range(len(X_all[actions[i]]))]

            row = [str(actions[i])] + row

            proposals_all.append(row)

            print("###")
            print("number =", i+1)
            print("actions = ", actions[i])
            print("proposal = ", X_all[actions[i]])
            print("###")


        with open(self.output_file, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(proposals_all)

        print("Finish selection of proposals!")

        return "True"
