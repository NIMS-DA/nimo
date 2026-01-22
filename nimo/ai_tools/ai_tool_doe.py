import numpy as np
import random
import copy
import csv
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from scipy.spatial.distance import cdist
from pyDOE3 import lhs


class DOE():
    """Class of DOE

    This class can select the next candidates by exhaustive search.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals, mode, max_iter):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for candidates for MI algorithm
            output_file (str): the file for proposals from MI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
            mode (str): "greedy", "distance", or "d-optimal"
            max_iter (int) : number of iteration in exchange method

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals
        self.mode = mode
        self.max_iter = max_iter

        if self.mode == None:
            self.mode = 'distance'

        if self.max_iter == None:
            self.max_iter = 1000


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

        This function is for RE.
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

        pool_array = X_all[test_actions]

        if len(train_actions) == 0:
            existing_array = []
        else:
            existing_array = X_all[train_actions]

        pool = np.array(pool_array)
        if len(existing_array) == 0:
            existing = np.empty((0, pool.shape[1]))
        else:
            existing = np.array(existing_array)

        scaler = MinMaxScaler(feature_range = (-1, 1))
        combined = np.vstack([pool, existing]) if existing.size > 0 else pool
        scaler.fit(combined)
    
        pool_scaled = scaler.transform(pool)
        existing_scaled = scaler.transform(existing) if existing.size > 0 else existing
   
        #score
        poly = PolynomialFeatures(degree = 2)
        def get_score(indices):
            current = pool_scaled[indices]
            if existing_scaled.size > 0:
                current = np.vstack([existing_scaled, current])
        
            if self.mode == "distance":
                if len(current) < 2: return 0
                d = pairwise_distances(current)
                np.fill_diagonal(d, np.inf)
                return np.min(d)

            if self.mode == "d_optimal":
                X = poly.fit_transform(current)
                if X.shape[0] < X.shape[1]: return -np.inf
                _, logdet = np.linalg.slogdet(X.T @ X)
                return logdet

        #greedy sampling
        if self.mode == "greedy":

            #Initial by Greedy sampling
            selected_indices = []
        
            if existing_scaled.size == 0:
                first_idx = np.random.choice(len(pool_scaled))
                selected_indices.append(first_idx)
                current_design = pool_scaled[first_idx].reshape(1, -1)
            else:
                current_design = existing_scaled

            while len(selected_indices) < self.num_proposals:
                dists = pairwise_distances(pool_scaled, current_design)
                min_dists = np.min(dists, axis = 1)
                
                for idx in selected_indices:
                    min_dists[idx] = - 1.0
            
                best_idx = np.argmax(min_dists)
                selected_indices.append(best_idx)
                current_design = np.vstack([current_design, pool_scaled[best_idx]])

        #exchange sampling
        if self.mode == "distance" or self.mode == "d_optimal":

            #Initial by Greedy sampling
            selected_indices = []
        
            if existing_scaled.size == 0:
                first_idx = np.random.choice(len(pool_scaled))
                selected_indices.append(first_idx)
                current_design = pool_scaled[first_idx].reshape(1, -1)
            else:
                current_design = existing_scaled

            while len(selected_indices) < self.num_proposals:
                dists = pairwise_distances(pool_scaled, current_design)
                min_dists = np.min(dists, axis = 1)
                
                for idx in selected_indices:
                    min_dists[idx] = - 1.0
            
                best_idx = np.argmax(min_dists)
                selected_indices.append(best_idx)
                current_design = np.vstack([current_design, pool_scaled[best_idx]])
        

            #Exchange Algorithm
            current_score = get_score(selected_indices)
            for _ in range(self.max_iter):
                idx_in_set = np.random.randint(0, len(selected_indices))
                idx_in_pool = np.random.randint(0, len(pool))
        
                if idx_in_pool in selected_indices: continue
        
                new_indices = list(selected_indices)
                new_indices[idx_in_set] = idx_in_pool
                new_score = get_score(new_indices)
        
                if new_score > current_score:
                    selected_indices = new_indices
                    current_score = new_score


        #LHS sampling
        if self.mode == "lhs":

            #ideal LHS generation
            total_needed = len(existing_scaled) + self.num_proposals
            ideal_lhs = lhs(len(pool_scaled[0]), samples = total_needed, criterion = 'maximin')

            scaler = MinMaxScaler(feature_range = (-1, 1))
            scaler.fit(ideal_lhs)
    
            ideal_lhs_scaled = scaler.transform(ideal_lhs)

            #Exclude points close to existing ones
            dists_to_existing = cdist(ideal_lhs_scaled, existing_scaled)
            used_lhs_indices = np.argmin(dists_to_existing, axis = 0)
            remaining_lhs = np.delete(ideal_lhs_scaled, used_lhs_indices, axis = 0)

            selected_indices = []
    
            for target in remaining_lhs:
                dists = cdist([target], pool_scaled)
                best_idx = np.argmin(dists)
       
                selected_indices.append(best_idx)

            selected_indices = list(set(selected_indices))

            #If duplicates exist, add them randomly
            if len(selected_indices) < self.num_proposals:
            
                num_needed = self.num_proposals - len(selected_indices)
                
                all_indices = set(range(len(pool_scaled)))
                candidates = list(all_indices - set(selected_indices))

                additional_idx = random.sample(candidates, min(len(candidates), num_needed))
                selected_indices.extend(additional_idx)

        actions = np.array(test_actions)[selected_indices]

        return actions


    def select(self):
        """Selecting the proposals by MI algorithm

        This function do not depend on robot.

        Returns:
            True (str) for success.

        """

        print("Start selection of proposals by DOE!")

        t_train, X_all, train_actions, test_actions = self.load_data()

        actions = self.calc_ai(t_train = t_train, X_all = X_all, 
        train_actions = train_actions, test_actions = test_actions)

        if len(actions) == 0:
            print("!!!There are no new proposals at this time!!!")
            import sys
            sys.exit()


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


