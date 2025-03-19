import sys
import os

import numpy as np
import random
import copy
import csv

import physbo
import itertools 


class PHYSBO():
    """Class of PHYSBO

    This class can select the next candidates by Bayesian optimization based on PHYSBO package.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals, physbo_score, minimization, ard, output_res):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for candidates for MI algorithm
            output_file (str): the file for proposals from MI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
            physbo_score (str): 'TS', 'EI', 'PI'
            ard (str) : True or False to perform ARD
            output_res (str): True or False to export prediction results

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals
        self.score = physbo_score
        self.minimization = minimization
        self.ard = ard
        self.output_res = output_res

        if self.score == None:
            self.score = 'TS'

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
        
        if self.minimization == False:
            t_train = arr_train[:, - self.num_objectives:]
        else:
            t_train = - arr_train[:, - self.num_objectives:]

        X_test = arr_test[:, : - self.num_objectives]

        test_actions = np.where(np.isnan(arr[:, -1]))[0].tolist()

        X_all=arr[:, : - self.num_objectives]

        all_actions = [i for i in range(len(X_all))]

        train_actions = np.sort(list(set(all_actions) - set(test_actions)))

        return t_train, X_all, train_actions, test_actions



    def calc_ai(self, t_train, X_all, train_actions, test_actions):
        """Calculating the proposals by AI algorithm

        This function is for PHYSBO.
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

        if self.ard == True:

            ##
            if self.num_objectives == 1:

                calculated_ids = train_actions

                t_initial = np.array( list(itertools.chain.from_iterable(t_train)) )

                X = physbo.misc.centering( X_all )
                
                policy = physbo.search.discrete.policy( test_X = X, initial_data = [calculated_ids, t_initial] )

                print("ard mode")

                from physbo.gp.predictor import predictor as Predictor
                from physbo.gp.cov.gauss import gauss as Gauss
                from physbo.gp.core import model as Model
                from physbo.gp.mean import const as Const
                from physbo.gp.lik import gauss as GaussLik
                from physbo.misc import set_config

                policy.predictor = Predictor(set_config(), Model(
                    cov=Gauss(len(X[0]), ard=True),
                    mean=Const(),
                    lik=GaussLik()
                ))

                policy.set_seed( 0 )

                actions = policy.bayes_search( max_num_probes = 1, num_search_each_probe = self.num_proposals, 
                simulator = None, score = self.score, interval = 0,  num_rand_basis = 1000 )

                #Output prediction results
                if self.output_res == True:

                    res_tot = []

                    f = open(self.input_file, 'r')
                    reader = csv.reader(f)
                    header = next(reader)

                    header.append('variance')
                    header.append('acquisition')

                    res_tot.append(header)

                    X_test = X[test_actions]
                    X_test_original = X_all[test_actions]

                    mean = policy.get_post_fmean(X_test)
                    var = policy.get_post_fcov(X_test)
                    score = policy.get_score(mode = self.score, xs = X_test)


                    for ii in range(len(X_test)):

                        res_each = []

                        for jj in range(len(X_test[0])):
                            res_each.append(X_test_original[ii][jj])

                        res_each.append(mean[ii])
                        res_each.append(var[ii])
                        res_each.append(score[ii])

                        res_tot.append(res_each)


                    with open('output_res.csv', 'w', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(res_tot)

                return actions


            ##
            if self.num_objectives > 1:

                calculated_ids = train_actions

                t_initial = np.array( t_train )

                X = physbo.misc.centering( X_all )

                policy = physbo.search.discrete_multi.policy( test_X = X, num_objectives = self.num_objectives,
                initial_data = [calculated_ids, t_initial])

                print("ard mode")
                
                from physbo.gp.predictor import predictor as Predictor
                from physbo.gp.cov.gauss import gauss as Gauss
                from physbo.gp.core import model as Model
                from physbo.gp.mean import const as Const
                from physbo.gp.lik import gauss as GaussLik
                from physbo.misc import set_config

                policy.predictor = Predictor(set_config(), Model(
                    cov=Gauss(len(X[0]), ard=True),
                    mean=Const(),
                    lik=GaussLik()
                ))

                policy.set_seed( 0 )

                actions = policy.bayes_search( max_num_probes = 1, num_search_each_probe = self.num_proposals, 
                simulator = None, score = self.score, interval = 0,  num_rand_basis = 500 )

                #Output prediction results
                if self.output_res == True:

                    res_tot = []

                    f = open(self.input_file, 'r')
                    reader = csv.reader(f)
                    header = next(reader)

                    for ii in range(self.num_objectives):
                        header.append('variance' + str(int(ii)))

                    res_tot.append(header)

                    X_test = X[test_actions]
                    X_test_original = X_all[test_actions]

                    mean = policy.get_post_fmean(X_test)
                    var = policy.get_post_fcov(X_test)

                    for ii in range(len(X_test)):

                        res_each = []

                        for jj in range(len(X_test[0])):
                            res_each.append(X_test_original[ii][jj])

                        for jj in range(self.num_objectives):
                            res_each.append(mean[ii][jj])
                        
                        for jj in range(self.num_objectives):
                            res_each.append(var[ii][jj])

                        res_tot.append(res_each)


                    with open('output_res.csv', 'w', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(res_tot)


            return actions


        else:

            ##
            if self.num_objectives == 1:

                calculated_ids = train_actions

                t_initial = np.array( list(itertools.chain.from_iterable(t_train)) )

                X = physbo.misc.centering( X_all )

                policy = physbo.search.discrete.policy( test_X = X, initial_data = [calculated_ids, t_initial] )

                policy.set_seed( 0 )

                actions = policy.bayes_search( max_num_probes = 1, num_search_each_probe = self.num_proposals, 
                simulator = None, score = self.score, interval = 0,  num_rand_basis = 1000 )

                
                #Output prediction results
                if self.output_res == True:

                    res_tot = []

                    f = open(self.input_file, 'r')
                    reader = csv.reader(f)
                    header = next(reader)

                    header.append('variance')
                    header.append('acquisition')

                    res_tot.append(header)

                    X_test = X[test_actions]
                    X_test_original = X_all[test_actions]

                    mean = policy.get_post_fmean(X_test)
                    var = policy.get_post_fcov(X_test)
                    score = policy.get_score(mode = self.score, xs = X_test)


                    for ii in range(len(X_test)):

                        res_each = []

                        for jj in range(len(X_test[0])):
                            res_each.append(X_test_original[ii][jj])

                        res_each.append(mean[ii])
                        res_each.append(var[ii])
                        res_each.append(score[ii])

                        res_tot.append(res_each)


                    with open('output_res.csv', 'w', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(res_tot)

                return actions



            ##
            if self.num_objectives > 1:

                calculated_ids = train_actions

                t_initial = np.array( t_train )

                X = physbo.misc.centering( X_all )

                policy = physbo.search.discrete_multi.policy( test_X = X, num_objectives = self.num_objectives,
                initial_data = [calculated_ids, t_initial])

                policy.set_seed( 0 )

                actions = policy.bayes_search( max_num_probes = 1, num_search_each_probe = self.num_proposals, 
                simulator = None, score = self.score, interval = 0,  num_rand_basis = 500 )


                #Output prediction results
                if self.output_res == True:

                    res_tot = []

                    f = open(self.input_file, 'r')
                    reader = csv.reader(f)
                    header = next(reader)

                    for ii in range(self.num_objectives):
                        header.append('variance' + str(int(ii)))

                    res_tot.append(header)

                    X_test = X[test_actions]
                    X_test_original = X_all[test_actions]

                    mean = policy.get_post_fmean(X_test)
                    var = policy.get_post_fcov(X_test)

                    for ii in range(len(X_test)):

                        res_each = []

                        for jj in range(len(X_test[0])):
                            res_each.append(X_test_original[ii][jj])

                        for jj in range(self.num_objectives):
                            res_each.append(mean[ii][jj])
                        
                        for jj in range(self.num_objectives):
                            res_each.append(var[ii][jj])

                        res_tot.append(res_each)


                    with open('output_res.csv', 'w', newline="") as f:
                        writer = csv.writer(f)
                        writer.writerows(res_tot)

                return actions



    def select(self):
        """Main function to select the proposals by AI algorithm

        This function do not depend on robot.

        Returns:
            True (str) for success.

        """

        print("Start selection of proposals by PHYSBO!")

        t_train, X_all, train_actions, test_actions = self.load_data()

        actions = self.calc_ai(t_train = t_train, X_all = X_all, 
        train_actions = train_actions, test_actions = test_actions)


        print('Proposals')

        proposals_all = []

        input_data = open(self.input_file, 'r')
        indexes = input_data.readlines()[0].rstrip('\n').split(',')

        indexes = ['actions'] + indexes[0 : - self.num_objectives]

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

        return 'True'
