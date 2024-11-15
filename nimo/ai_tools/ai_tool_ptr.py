import numpy as np
import random
import copy
import csv
import physbo
from scipy.stats import norm

class PTR():
    """Class of PTR

    This class can select the next candidates by random exploration.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals, ptr_ranges, output_res):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for candidates for MI algorithm
            output_file (str): the file for proposals from MI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
            ptr_ranges (list): the ranges for PTR
            output_res (str): True or False to export prediction results

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals
        self.ranges = ptr_ranges
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

        X = np.array(X_all)
        X_train = np.array(X[train_actions])
        X_test = np.array(X[test_actions])

        actions = []

        max_list = np.max(t_train, axis=0)

        min_list = np.min(t_train, axis=0)


        for jj in range(len(self.ranges)):
            for ii in range(len(self.ranges[0])):

                if self.ranges[jj][ii] == "max":
                    self.ranges[jj][ii] = max_list[jj]
                if self.ranges[jj][ii] == "min":
                    self.ranges[jj][ii] = min_list[jj]


        for kk in range(self.num_proposals):

            tot_mean = []
            tot_std = []

            for ii in range(len(t_train[0])):

                y_train = np.array([r[ii] for r in t_train])

                cov  = physbo.gp.cov.gauss( X_train.shape[1], ard = True )
                mean = physbo.gp.mean.const()
                lik  = physbo.gp.lik.gauss()
                gp = physbo.gp.model( lik=lik, mean=mean, cov=cov )
                config = physbo.misc.set_config()

                gp.fit( X_train, y_train, config )

                gp.prepare( X_train, y_train )
                y_mean = gp.get_post_fmean( X_train, X_test )
                y_cov = gp.get_post_fcov( X_train, X_test )

                tot_mean.append(y_mean)
                tot_std.append([np.sqrt(y_cov[i]) for i in range(len(y_cov))])


            scores = []
            cdfs_all = []

            for ii in range(len(tot_mean[0])):

                cdfs = []

                for jj in range(len(tot_mean)):

                    cdfs.append(norm.cdf((self.ranges[jj][1] - tot_mean[jj][ii])/tot_std[jj][ii])
                    - norm.cdf((self.ranges[jj][0] - tot_mean[jj][ii])/tot_std[jj][ii]))

                cdfs_all.append(cdfs)

                prob = 1

                for jj in range(len(cdfs)):

                    prob = prob * cdfs[jj]

                scores.append(prob)

            index = np.array(scores).argsort()[::-1]


            #Output prediction results
            if kk == 0 and self.output_res == True:

                res_tot = []

                f = open(self.input_file, 'r')
                reader = csv.reader(f)
                header = next(reader)

                header.append('acquisition')

                res_tot.append(header)

                for ii in range(len(X_test)):

                    res_each = []

                    for jj in range(len(X_test[0])):
                        res_each.append(X_test[ii][jj])

                    for jj in range(len(tot_mean)):
                        res_each.append(tot_mean[jj][ii])
      
                    res_each.append(scores[ii])

                    res_tot.append(res_each)


                with open('output_res.csv', 'w', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(res_tot)


            #select candidates
            actions.append(test_actions[index[0]])
            
            t_vals = [tot_mean[i][index[0]] for i in range(len(tot_mean))]

            popped_item = test_actions.pop(index[0])
            train_actions = np.sort(np.insert(train_actions, -1, popped_item))
        
            X_train = np.array(X[train_actions])
            X_test = np.array(X[test_actions])

            t_train = np.insert(t_train, list(train_actions).index(popped_item), t_vals, axis = 0)


        return actions



    def select(self):
        """Selecting the proposals by MI algorithm

        This function do not depend on robot.

        Returns:
            True (str) for success.

        """

        print("Start selection of proposals by PTR!")

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
