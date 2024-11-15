import numpy as np
import random
import copy
import csv
import physbo
import itertools
import sys
import os

class SLESA():
    """Class of SLESA

    This class can select the next candidates by random exploration.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals, slesa_beta_max, slesa_beta_num, re_seed, output_res):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for candidates for MI algorithm
            output_file (str): the file for proposals from MI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
            slesa_beta_max (float): the maximum value of beta (inverse temperature)
            slesa_beta_num (int): the number of beta (number of cycles)
            re_seed (int): seed of random number
            output_res (str): True or False to export prediction results


        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals

        self.beta_max = slesa_beta_max
        self.beta_num = slesa_beta_num
        self.seed = re_seed

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

        betas = np.linspace(0,self.beta_max,self.beta_num)

        path = 'slesa_log_index.csv'
        is_file = os.path.isfile(path)

        if is_file:

            with open('slesa_log_index.csv') as f:
                reader = csv.reader(f)
                l = [row for row in reader]

            #min-max standardization by initial data
            E_stand = []

            try:
                for ii in range(len(l)-1):
                    index = list(train_actions).index(int(l[ii+1][0]))

                    E_stand.append(list(itertools.chain.from_iterable(t_train))[index])

            except:
                print('WARNINGS from SLESA module')
                print('The incorrect log file might be used.')
                sys.exit()

            E_min = min(E_stand)
            E_max = max(E_stand)

            #training data
            biter = len(l[0])

            if len(l[0]) >= self.beta_num:
                print("SLESA calculations were already finished!")
                sys.exit()

            X = physbo.misc.centering( np.array(X_all) )
            X_train = np.array(X[train_actions])

            #small objective function is large energy
            y_train = np.array([(E_max - t_train[i][0])/(E_max - E_min) for i in range(len(t_train))])

            cov  = physbo.gp.cov.gauss( X_train.shape[1], ard = True )
            mean = physbo.gp.mean.const()
            lik  = physbo.gp.lik.gauss()
            gp = physbo.gp.model( lik=lik, mean=mean, cov=cov )
            config = physbo.misc.set_config()

            gp.fit( X_train, y_train, config )

            gp.prepare( X_train, y_train )
            y_mean = gp.get_post_fmean( X_train, X )

            #sampling
            prob = np.exp(- betas[biter] * y_mean)
            prob = prob/np.sum(prob)

            id = np.random.choice([i for i in range(len(prob))], self.num_proposals, p = prob, replace = True)


            #update log file
            new_l = []

            for ii in range(len(l)):

                l_add = l[ii]

                if ii == 0:
                    l_add.append(betas[biter])

                else:
                    l_add.append(id[ii-1])

                new_l.append(l_add)

            with open('slesa_log_index.csv', 'w', newline = "") as f:
                writer = csv.writer(f)
                writer.writerows(new_l)


            if len(l[0]) == 2:

                #create pred energy file
                new_e = []

                for ii in range(len(l)):

                    e_add = []

                    if ii == 0:
                        e_add.append(betas[biter])

                    else:
                        e_add.append(y_mean[id[ii-1]])

                    new_e.append(e_add)

                with open('slesa_log_e.csv', 'w', newline = "") as f:
                    writer = csv.writer(f)
                    writer.writerows(new_e)

            else:

                with open('slesa_log_e.csv') as f:
                    reader = csv.reader(f)
                    e = [row for row in reader]

                #update pred energy file
                new_e = []

                for ii in range(len(l)):

                    e_add = e[ii]

                    if ii == 0:
                        e_add.append(betas[biter])

                    else:
                        e_add.append(y_mean[id[ii-1]])

                    new_e.append(e_add)

                with open('slesa_log_e.csv', 'w', newline = "") as f:
                    writer = csv.writer(f)
                    writer.writerows(new_e)


            #next actions
            actions = []

            for ii in range(len(id)):

                if id[ii] not in train_actions:
                    if id[ii] not in actions:
                        actions.append(id[ii])


            #Output prediction results
            if self.output_res == True:

                res_tot = []

                f = open(self.input_file, 'r')
                reader = csv.reader(f)
                header = next(reader)

                header.append('variance')

                res_tot.append(header)

                X_test = X[test_actions]
                X_test_original = X_all[test_actions]

                mean = gp.get_post_fmean(X_train, X_test)
                var = gp.get_post_fcov(X_train, X_test)


                for ii in range(len(X_test)):

                    res_each = []

                    for jj in range(len(X_test[0])):
                        res_each.append(X_test_original[ii][jj])

                    res_each.append(mean[ii])
                        
                    res_each.append(var[ii])

                    res_tot.append(res_each)


                with open('output_res.csv', 'w', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(res_tot)
        
        else:
            
            if self.seed  != None:
                random.seed(self.seed) 

            actions = random.sample(test_actions, self.num_proposals)
            log = [[betas[0]]]

            for ii in range(len(actions)):
                log.append([actions[ii]])

            with open('slesa_log_index.csv', 'w', newline = "") as f:
                writer = csv.writer(f)
                writer.writerows(log)

        return actions



    def select(self):
        """Selecting the proposals by MI algorithm

        This function do not depend on robot.

        Returns:
            True (str) for success.

        """

        print("Start selection of proposals by SLESA!")

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
