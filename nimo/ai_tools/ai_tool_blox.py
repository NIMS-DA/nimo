import numpy as np
import random
import copy
import csv

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


class BLOX():
    """Class of BLOX

    This class can select the next candidates by random exploration.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals, output_res):
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
        
        self.output_res = output_res


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

        This function is for BLOX.
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

        prediction_model = "RF"

        #train prediction models
        def build_model(prediction_model, x_train, y_train):
            """Building prediction model

            This function do not depend on robot.

            Args:
                prediction_model (str): "RF"
                x_train (list[float]): the list of training features
                y_train (list[float]): the list of training labels

            Returns:
                model: the prediction model

            """

            if prediction_model == 'RF': 
                params = {'n_estimators':[10, 50, 100]}
                gridsearch = GridSearchCV(RandomForestRegressor(), param_grid = params, cv = 3, scoring = "r2", n_jobs = 1, verbose = 1)
                gridsearch.fit(x_train,y_train)
                model =  RandomForestRegressor(n_estimators = gridsearch.best_params_['n_estimators'])
                model.fit(x_train, y_train)
                return model


        #stein novelty calculation
        def hesgau(x, y, sigma):
            """Calculating hesgau

            This function do not depend on robot.

            Args:
                x (list[float]): the list of target 1
                y (list[float]): the list of target 2
                sigma (int): the variance of Gaussian distribution

            Returns:
                the value of hesgau

            """

            dim = len(x)
            dist = np.sum(np.power(x - y, 2))
            return (dim / sigma - dist / sigma**2) * np.exp(- dist / (2*sigma))


        def stein_novelty(point, data_list, sigma):
            """Calculating stein_novelty

            This function do not depend on robot.

            Args:
                point (list[float]): the list of testing data
                data_list (list[float]): the list of training data
                sigma (int): the variance of Gaussian distribution

            Returns:
                the score of stein novelty

            """

            n = len(data_list)
            score = 0
            score = np.sum([hesgau(point, data_list[k,:], sigma) for k in range(n)])
            score = score / (n * (n + 1) /2)
            return - score



        #Preparation of data    
        features_observed = X_all[train_actions]
        features_unchecked = X_all[test_actions]
        properties_observed = t_train


        sc = StandardScaler()
        sc.fit(features_observed)
        sc_features_observed = sc.transform(features_observed)
        sc_features_unchecked = sc.transform(features_unchecked)

        sc_property = StandardScaler()
        sc_property.fit(properties_observed)

        dimension = self.num_objectives


        actions = []


        for i in range(self.num_proposals):

            sc_properties_observed = sc_property.transform(properties_observed)

            #Build prediction models
            model_list = []
            for d in range(dimension):
                model = build_model(prediction_model, sc_features_observed, properties_observed[:,d])
                model_list.append(model)


            #Predict properties of unchecked data
            predicted_properties_list = []
            for d in range(dimension):
                predicted_properties_list.append(model_list[d].predict(sc_features_unchecked))
            predicted_properties_list = np.array(predicted_properties_list).T

            
            #Calc. Stein Novelty
            sc_predicted_properties_list = sc_property.transform(predicted_properties_list) 
            sn_data = [stein_novelty(point, sc_properties_observed, sigma=0.1) for point in sc_predicted_properties_list]


            #Output prediction results
            if i == 0 and self.output_res == True:
        
                res_tot = []

                f = open(self.input_file, 'r')
                reader = csv.reader(f)
                header = next(reader)

                header.append('acquisition')

                res_tot.append(header)

                for ii in range(len(features_unchecked)):

                    res_each = []

                    for jj in range(len(features_unchecked[0])):
                        res_each.append(features_unchecked[ii][jj])

                    for jj in range(len(predicted_properties_list[0])):
                        res_each.append(predicted_properties_list[ii][jj])
          
                    res_each.append(sn_data[ii])

                    res_tot.append(res_each)


                with open('output_res.csv', 'w', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(res_tot)

            
            #Select next candidates

            rank_index = np.array(sn_data).argsort()[::-1]

            actions.append(test_actions[rank_index[0]])
           

            sc_features_observed = sc_features_observed.tolist()
            properties_observed =  properties_observed.tolist()

            sc_features_observed.append(sc_features_unchecked[rank_index[0]])
            properties_observed.append(predicted_properties_list[rank_index[0]])

            sc_features_observed = np.array(sc_features_observed)
            properties_observed = np.array(properties_observed)


            test_actions.pop(rank_index[0])

            sc_features_unchecked = np.delete(sc_features_unchecked, rank_index[0], axis = 0)

        return actions



    def select(self):
        """Selecting the proposals by MI algorithm

        This function do not depend on robot.

        Returns:
            True (str) for success.

        """

        print("Start selection of proposals by BLOX!")

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
