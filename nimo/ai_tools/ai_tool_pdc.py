import numpy as np
import random
import copy
import csv
from scipy import stats

import sklearn.semi_supervised
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


class PDC():
    """Class of PDC

    This class can select the next candidates by phase diagram construction.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals, pdc_estimation, pdc_sampling, output_res):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for candidates for MI algorithm
            output_file (str): the file for proposals from MI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
            pdc_estimation (str): estimation methods: 'LP' or 'LS'
            pdc_sampling (str): sampling methods: 'LC' ,'MS', 'EA'
            output_res (str): True or False to export prediction results

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals

        self.LP_algorithm = pdc_estimation
        self.US_strategy = pdc_sampling
        self.output_res = output_res
        
        if self.LP_algorithm == None:
            self.LP_algorithm = 'LP'
        if self.US_strategy == None:
            self.US_strategy = 'LC'


    def load_data(self):
        """Loading candidates

        This function do not depend on robot.

        Returns:
            t_train (list[float]): the list where observed objectives are stored
            X_all (list[float]): the list where all descriptors are stored
            train_actions (list[float]): the list where observed actions are stored
            test_actions (list[float]): the list where test actions are stored

        """

        if self.num_objectives != 1:
            raise ValueError("PDC requires exactly one classification label column")
        with open(self.input_file, newline='', encoding='utf-8-sig') as f:
            rows = list(csv.reader(f))
        if not rows or len(rows[0]) < 2:
            raise ValueError("CSV requires numeric features and one label column")
        header, rows = rows[0], rows[1:]
        if not rows or any(len(row) != len(header) for row in rows):
            raise ValueError("CSV must contain nonempty, rectangular candidate data")
        X_all = np.asarray([[float(v) for v in row[:-1]] for row in rows])
        if not np.isfinite(X_all).all():
            raise ValueError("PDC features must be finite numbers")
        labels = [row[-1].strip() for row in rows]
        train_actions = [i for i, label in enumerate(labels) if label]
        test_actions = [i for i, label in enumerate(labels) if not label]
        t_train = np.asarray([[labels[i]] for i in train_actions], dtype=object)

        return t_train, X_all, train_actions, test_actions



    def calc_ai(self, t_train, X_all, train_actions, test_actions):
        """Calculating the proposals by AI algorithm

        This function is for PDC.
        This function do not depend on robot.
        If the new AI algorithm is developed, this function is only changed.
        
        Args:
            t_train (list[float]): the list where observed objectives are stored
            X_all (list[float]): the list where all descriptors are stored
            train_actions (list[float]): the list where observed actions are stored
            test_actions (list[float]): the list where test actions are stored

        Returns:
            actions (list[int]): the list where the selected actions are stored

        """

        #parameters
        #LP_algorithm = 'LP' #'LP', 'LS'                                                                   
        #US_strategy = 'LC'  #'LC' ,'MS', 'EA', 'RS'
        multi_method = 'OU' #'OU', 'NE'
        k = 2


        #Preparation of data    
        data_list = np.array(X_all)

        # Encode categories; -1 is reserved internally for unobserved rows.
        observed = [str(row[0]).strip() for row in t_train]
        self.classes_ = sorted(set(observed))
        # Preserve the numerical class ordering of legacy integer-labelled data.
        try:
            numeric = {label: float(label) for label in self.classes_}
            if all(np.isfinite(v) for v in numeric.values()):
                self.classes_.sort(key=lambda label: (numeric[label], label))
        except ValueError:
            pass
        if len(self.classes_) < 2:
            raise ValueError("PDC requires at least two observed classes; use RE for initialization")
        if not test_actions:
            raise ValueError("PDC requires at least one unobserved candidate")
        if self.num_proposals < 1 or self.num_proposals > len(test_actions):
            raise ValueError("num_proposals must be between 1 and the unobserved candidate count")
        mapping = {label: i for i, label in enumerate(self.classes_)}
        label_list = np.full(len(data_list), -1, dtype=int)
        for action, label in zip(train_actions, observed):
            label_list[action] = mapping[label]

        label_index_list = range(len(data_list))
        labeled_index_list = train_actions
        unlabeled_index_list = test_actions
        dimension = len(data_list[0])

            
        ss = StandardScaler()
        ss.fit(data_list)
        data_list_std = ss.transform(data_list)



        #----SAMPLING
        label_train = np.copy(label_list)

        #estimate phase of each point
        if self.LP_algorithm == 'LS':
            #lp_model = label_propagation.LabelSpreading()
            lp_model = sklearn.semi_supervised.LabelSpreading()
        elif self.LP_algorithm == 'LP':
            #lp_model = label_propagation.LabelPropagation()
            lp_model = sklearn.semi_supervised.LabelPropagation()

        lp_model.fit(data_list_std, label_train)
        predicted_labels = lp_model.transduction_[unlabeled_index_list]
        predicted_all_labels = lp_model.transduction_
        label_distributions = lp_model.label_distributions_[unlabeled_index_list]
        label_distributions_all = lp_model.label_distributions_
        classes = lp_model.classes_


        # probability_j corresponds to classes_[j], also saved beside predictions.
        self.predicted_labels_ = np.asarray(self.classes_)[predicted_all_labels]
        self.label_distributions_ = label_distributions_all

        #Output prediction results
        if self.output_res == True:

            res_tot = []

            f = open(self.input_file, 'r')
            reader = csv.reader(f)
            header = next(reader)

            for jj in range(len(label_distributions_all[0])):

                header.append('probability_' + str(jj))

            res_tot.append(header)

            for ii in range(len(X_all)):

                res_each = []

                for jj in range(len(X_all[0])):
                    res_each.append(X_all[ii][jj])

                res_each.append(self.classes_[predicted_all_labels[ii]])
                
                for jj in range(len(label_distributions_all[0])):
                    res_each.append(label_distributions_all[ii][jj])

                res_tot.append(res_each)


            with open('output_res_classes.csv', 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['probability_column', 'label'])
                writer.writerows(('probability_' + str(i), label)
                                 for i, label in enumerate(self.classes_))

            with open('output_res.csv', 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(res_tot)



        #calculate Uncertainly Score
        if self.US_strategy == 'EA':
            pred_entropies = stats.distributions.entropy(label_distributions.T)
            u_score_list = pred_entropies/np.max(pred_entropies)
            
            uncertainty_index = [unlabeled_index_list[np.argmax(u_score_list)]]

            #############
            # all ranking of uncertain point
            ranking = np.array(u_score_list).argsort()[::-1]
            multi_uncertainty_index = [unlabeled_index_list[ranking[i]] for i in range(len(unlabeled_index_list))]
            #############

        elif self.US_strategy == 'LC':
            u_score_list = 1 - np.max(label_distributions, axis = 1)
            uncertainty_index = [unlabeled_index_list[np.argmax(u_score_list)]]
                    
            #############
            # all ranking of uncertain point
            ranking = np.array(u_score_list).argsort()[::-1]
            multi_uncertainty_index = [unlabeled_index_list[ranking[i]] for i in range(len(unlabeled_index_list))]
            #############
            

        elif self.US_strategy == 'MS':

            u_score_list = []
            for pro_dist in label_distributions:
                pro_ordered = np.sort(pro_dist)[::-1]
                margin = pro_ordered[0] - pro_ordered[1]
                u_score_list.append(margin)
            
            u_score_list = 1 - np.array(u_score_list)

            uncertainty_index = [unlabeled_index_list[np.argmax(u_score_list)]]
                    
            #############
            # all ranking of uncertain point
            ranking = np.array(u_score_list).argsort()[::-1]
            multi_uncertainty_index = [unlabeled_index_list[ranking[i]] for i in range(len(unlabeled_index_list))]
            #############


        if self.num_proposals == 1:

            actions = [uncertainty_index[0]]


        else:

            #############
            #multi

            if multi_method == 'OU':

                US_point = multi_uncertainty_index[0:self.num_proposals]

            if multi_method == 'NE':

                from scipy.spatial import distance

                neighbor_dist = []

                for i in range(len(data_list)):

                    dist_value = round(distance.euclidean(data_list[0],data_list[i]),5)

                    if dist_value not in neighbor_dist:
                        neighbor_dist.append(dist_value)

                neighbor_dist.sort()

                delta = neighbor_dist[k]

                US_point = []

                for i in range(len(multi_uncertainty_index)):

                    if i == 0: US_point.append(multi_uncertainty_index[i])

                    true_num = 0

                    for j in range(len(US_point)):

                        two_dist = distance.euclidean(data_list[US_point[j]], data_list[multi_uncertainty_index[i]])

                        if round(two_dist, 5) > delta:
                                        
                            true_num += 1

                    if true_num == len(US_point) and len(US_point) < self.num_proposals: US_point.append(multi_uncertainty_index[i])


            actions = US_point

        return actions



    def select(self):
        """Selecting the proposals by MI algorithm

        This function do not depend on robot.

        Returns:
            True (str) for success.

        """

        print("Start selection of proposals by PDC!")

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
