import numpy as np
import physbo
import itertools
import csv
import enum


def dpp_mcmc(
    kernel_matrix,
    unnorm_prob,
    initial_batch,
    sigma: float = 0.01,
    batch_size: int = 10,
    mcmc_length: int = 200,
):
    """
    Returns a batch of next actions using DPP.

    Args:
        kernel_matrix : Kernel matrix that represents repulsions.
        unnorm_prob : Unnormalized probability.
        initial_batch : Initial batch of actions.
        sigma (float, optional): Sigma for DPP. Defaults to 0.01.
        batch_size (int, optional): Batch size. Defaults to 10.
        mcmc_length (int, optional): Length of MCMC for DPP. Defaults to 200.

    Returns:
        Batch of next actions.
    """
    current_batch = np.copy(initial_batch)
    for _ in range(mcmc_length):
        prob = np.copy(unnorm_prob)
        prob[current_batch] = 0
        prob = prob / np.sum(prob)
        K = kernel_matrix[np.ix_(current_batch, current_batch)]
        replace_candidate_idx = np.random.choice(list(range(batch_size)))
        next_candidate = np.random.choice(list(range(len(prob))), p=prob)
        next_batch = np.copy(current_batch)
        next_batch[replace_candidate_idx] = next_candidate
        K_tmp = kernel_matrix[np.ix_(next_batch, next_batch)]
        identity_matrix = np.eye(batch_size, batch_size)
        Lt = identity_matrix + K / sigma**2
        Lt_tmp = identity_matrix + K_tmp / sigma**2
        accept_prob = min(1, np.linalg.det(Lt_tmp) / np.linalg.det(Lt))
        if np.random.uniform(0, 1) < accept_prob:
            current_batch = next_batch
    return current_batch


#class Mode(enum.Enum):
#    """Enum class for the mode of threshold update
#    """
#    aggressive = 1
#    moderate = 2
#    conservative = 3

class NTS():
    """Class of NTS

    This class can select the next candidates by random exploration.

    """

    def __init__(self, input_file, output_file, num_objectives, num_proposals, sample_mode, minimization, use_dpp, re_seed, output_res):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for candidates for MI algorithm
            output_file (str): the file for proposals from MI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
            sample_mode (Mode): the mode of sampling
            minimization (str): True or False to perform minimization
            use_dpp (bool): whether to use DPP or not
            re_seed (int): seed of random number

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = 1
        self.num_proposals = num_proposals
        self.sample_mode = sample_mode
        if self.sample_mode == "aggressive":
            self.lstar_scale = 0.99
        elif self.sample_mode == "moderate":
            self.lstar_scale = 0.95
        elif self.sample_mode == "conservative":
            self.lstar_scale = 0.9
        else:
            raise ValueError("Invalid sample_mode")
        self.use_dpp = use_dpp
        self.seed = re_seed
        self.output_res = output_res
        # fixed parameters
        self.num_thompson_sampling = 100
        self.dpp_mcmc_length = num_proposals * 10
        self.dpp_sigma = 0.1
        
        self.minimization = minimization
        
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
        X = physbo.misc.centering( X_all )
        t_initial = np.array( list(itertools.chain.from_iterable(t_train)) )
        policy = physbo.search.discrete.Policy( test_X = X, initial_data = [train_actions, t_initial] )
        policy.bayes_search(
            max_num_probes=0,
            simulator=None,
            score="TS",
            interval=0,
            num_rand_basis=5000,
            is_disp=False,
        )
        # Threshold is a scaled value of the maximum value of the observed objectives
        lstar = np.percentile(t_initial, [int(self.lstar_scale * 100)])[0]
        print("lstar:", lstar)
        # Initialization by ones_like method means that each action has a chance to be selected
        hist = np.ones_like(test_actions) / 10000
        X_test = X[test_actions]
        for _ in range(self.num_thompson_sampling):
            sample = policy.get_score(mode="TS", xs=X_test)
            indicator_arr = np.where(sample > lstar, 1, 0)
            hist += indicator_arr
            if sum(indicator_arr) == 0:
                print("reducing lstar")
                if lstar > 0:
                    lstar *= 0.9
                elif lstar < 0:
                    lstar *= 1.1
                else:
                    lstar += 0.1
        action_idx_list = np.random.choice(
            np.arange(len(test_actions)), self.num_proposals, p=hist / sum(hist), replace=False
        )
        if self.use_dpp is True:
            post_kernel_matrix = self.policy.get_post_fcov(X_test, diag=False)
            action_idx_list = dpp_mcmc(
                post_kernel_matrix,
                hist / sum(hist),
                action_idx_list,
                sigma=self.dpp_sigma,
                batch_size=self.num_proposals,
                mcmc_length=self.dpp_mcmc_length,
            )
        
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
            score = policy.get_score(mode = "TS", xs = X_test)


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

        
        return list(np.array(test_actions)[action_idx_list])


    def select(self):
        """Selecting the proposals by MI algorithm

        This function do not depend on robot.

        Returns:
            True (str) for success.

        """

        print("Start selection of proposals by NTS!")

        t_train, X_all, train_actions, test_actions = self.load_data()

        actions = self.calc_ai(t_train = t_train, X_all = X_all, 
        train_actions = train_actions, test_actions = test_actions)

        #print("Selected actions:", actions)
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


