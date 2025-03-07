import sys
import os

import numpy as np
import random
import copy
import csv

import physbo
import itertools 


class COMBI():
    """Class of BOCOMBI

    This class can select the next candidates by Bayesian optimization based on PHYSBO package.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals, physbo_score, combi_ranges, spread_elements):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for candidates for MI algorithm
            output_file (str): the file for proposals from MI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
            physbo_score (str): the score of physbo
            combi_ranges (list[float]): the ranges for each element
            spread_elements (list[int]): the list of spread elements

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals
        self.score = physbo_score

        if self.score is None:
            self.score = 'TS'

        self.combi_ranges = combi_ranges
        self.spread_elements = spread_elements


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

        This function is for PHYSBO with inclination.
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

        #Best candidates
        calculated_ids = train_actions

        t_initial = np.array( list(itertools.chain.from_iterable(t_train)) )

        from sklearn import preprocessing
        ss = preprocessing.StandardScaler()
        X = ss.fit_transform(X_all)


        if self.score == "RE":
       
            if len(t_initial) == 0:
                policy = physbo.search.discrete.policy( test_X = X )
            else:
                policy = physbo.search.discrete.policy( test_X = X, initial_data = [calculated_ids, t_initial] )

            actions = policy.random_search(max_num_probes=1, simulator=None)

        else:

            policy = physbo.search.discrete.policy( test_X = X, initial_data = [calculated_ids, t_initial] )
            policy.set_seed( 0 )
            actions = policy.bayes_search( max_num_probes = 1, num_search_each_probe = 1, 
                    simulator = None, score = self.score, interval = 0,  num_rand_basis = 1000 )

        best_action = actions[0]

        best = X_all[best_action]


        print("###")
        print("number =", 1)
        print("actions = ", actions[0])
        print("proposal = ", X_all[actions[0]])
        print("###")


        # Create candidates data space
        inclination_elements = self.spread_elements
        
        del_list = []
        for ii in range(len(best)):
            if best[ii] == 0:

                for jj in range(len(inclination_elements)):
                    if ii in inclination_elements[jj]:
                        del_list.append(jj)


        remain_list = np.sort(list(set([ii for ii in range(len(best))]) - set(del_list)))

        inclination_elements = np.array(inclination_elements)[remain_list].tolist()


        cand_processes = []
        red_inclination_elements = []

        for ii in range(len(inclination_elements)):

            fix = [i for i in range(len(best))]
            fix.remove(inclination_elements[ii][0])
            fix.remove(inclination_elements[ii][1])

            remain = 100 - sum([best[i] for i in fix])

            list1 = list(range(self.combi_ranges[inclination_elements[ii][0]][0], self.combi_ranges[inclination_elements[ii][0]][1] + 1, 1))
            list2 = list(range(self.combi_ranges[inclination_elements[ii][1]][0], self.combi_ranges[inclination_elements[ii][1]][1] + 1, 1))

            ranges = []
            for jj1 in range(len(list1)):
                for jj2 in range(len(list2)):

                    if list1[jj1] + list2[jj2] == remain and list1[jj1] != 0 and list2[jj2] != 0:
                        ranges.append(list1[jj1])


            if len(ranges) >= 2:
                red_inclination_elements.append(inclination_elements[ii])
                cands = [round(min(ranges) + (max(ranges) - min(ranges))/(self.num_proposals - 1) * i, 2) for i in range(self.num_proposals)]


                for jj in range(self.num_proposals):
                    process = []

                    for kk in range(len(best)):
                        if kk in fix:
                            process.append(best[kk])
                        elif kk == inclination_elements[ii][0]:
                            process.append(cands[jj])
                        elif kk == inclination_elements[ii][1]:
                            process.append(remain - cands[jj])

                    cand_processes.append(process)

        if len(red_inclination_elements) == 0:
            print("!!!!Impossible to perform composition-spread!!!!")
        
            import sys
            sys.exit()


        if self.score == "RE":

            inclination_index = random.randrange(len(red_inclination_elements))

        else:

            acquisitions = policy.get_score(mode = self.score, xs = ss.transform(cand_processes))

            av_acquisitions = []


            for ii in range(len(red_inclination_elements)):

                av_acquisitions.append(np.mean(acquisitions[0 + ii * self.num_proposals : self.num_proposals + ii * self.num_proposals]))

            inclination_index = np.argmax(av_acquisitions)
        

        f = open(self.input_file, 'r')
        reader = csv.reader(f)
        header = next(reader)

        elements = []
        for ii in range(len(header) - self.num_objectives):
            elements.append(header[ii])

        print("Elements for composition-spread")
        print(elements[red_inclination_elements[inclination_index][0]], elements[red_inclination_elements[inclination_index][1]])


        f = open(self.input_file, 'r')
        reader = csv.reader(f)
        header = next(reader)

        proposal_data = [['actions'] + elements]

        for ii in range(self.num_proposals):

            proposal_data.append([-1] + list(cand_processes[inclination_index * self.num_proposals + ii]))


        with open(self.output_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(proposal_data)


        return best_action




    def select(self):
        """Main function to select the proposals by AI algorithm

        This function do not depend on robot.

        Returns:
            True (str) for success.

        """

        print("Start selection of proposals by COMBI!")

        t_train, X_all, train_actions, test_actions = self.load_data()

        actions = self.calc_ai(t_train = t_train, X_all = X_all, 
        train_actions = train_actions, test_actions = test_actions)

        
        print("Finish selection of proposals!")

        return 'True'
