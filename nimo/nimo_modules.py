import numpy as np
import nimo.ai_tools
import nimo.input_tools
import nimo.output_tools

class selection():
    """Class of selection

    This class can select the next candidates depending on the AI methods.

    """

    def __init__(self, method, input_file, output_file, num_objectives, num_proposals, 
                 re_seed = None,
                 ptr_ranges = None,
                 slesa_beta_max = None, slesa_beta_num = None,
                 physbo_score = None, minimization = None, ard = None,
                 pdc_estimation = None, pdc_sampling = None,
                 process_X = None,
                 combi_ranges = None, spread_elements = None,
                 output_res = None):

        """Constructor
        
        This function do not depend on robot.

        Args:
            method (str): "RE" or "BO"or "BLOX" or "PDC"
            input_file (str): the file for candidates for AI algorithm
            output_file (str): the file for proposals from AI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals
            re_seed (int): the value of seed
            ptr_ranges (list): the target ranges in PTR method
            slesa_beta_max (float): the value of beta max in SLESA method
            slesa_beta_num (int): the number of beta in SLESA method
            physbo_score (str): the acquisition function in PHYSBO method
            minimization (str): True or False for minimize or maximize
            ard (str): True or False to use ard mode in PHYSBO method
            pdc_estimation (str): estimation methods: 'LP' or 'LS' in PDC method
            pdc_sampling (str): sampling methods: 'LC' ,'MS', 'EA' in PDC method
            process_X (list) : index for process parameters in BOMP method
            combi_ranges (list[float]): the ranges for each element in COMBI method
            spread_elements (list[int]): the list of spread elements in COMBI method
            output_res (str): True or False to output res file

        """

        self.method = method
        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.num_proposals = num_proposals

        self.re_seed = re_seed
        self.ptr_ranges = ptr_ranges
        self.slesa_beta_max = slesa_beta_max
        self.slesa_beta_num = slesa_beta_num
        self.physbo_score = physbo_score
        self.minimization = minimization
        self.ard = ard
        self.pdc_estimation = pdc_estimation
        self.pdc_sampling = pdc_sampling
        
        self.process_X = process_X

        self.combi_ranges = combi_ranges
        self.spread_elements = spread_elements

        self.output_res = output_res

        res = self.module_selection()
        


    def module_selection(self):
        """module selection of preparation input
        
        This function do not depend on robot.

        Returns:
            res (str): True for success, False otherwise.

        """
        res = 'False'
        if self.method == "RE":
            res = nimo.ai_tools.ai_tool_re.RE(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.process_X, self.re_seed).select()
            return res

        if self.method == "PHYSBO":
            res = nimo.ai_tools.ai_tool_physbo.PHYSBO(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.physbo_score, self.minimization, self.ard, self.output_res).select()
            return res

        if self.method == "PDC":
            res = nimo.ai_tools.ai_tool_pdc.PDC(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.pdc_estimation, self.pdc_sampling,
            self.output_res).select()
            return res

        if self.method == "BLOX":
            res = nimo.ai_tools.ai_tool_blox.BLOX(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.output_res).select()
            return res
        
        if self.method == "PTR":
            res = nimo.ai_tools.ai_tool_ptr.PTR(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.ptr_ranges, self.output_res).select()
            return res

        if self.method == "SLESA":
            res = nimo.ai_tools.ai_tool_slesa.SLESA(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.slesa_beta_max, self.slesa_beta_num, 
            self.re_seed, self.output_res).select()
            return res
        
        if self.method == "BOMP":
            res = nimo.ai_tools.ai_tool_bomp.BOMP(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.physbo_score, self.minimization, self.process_X, self.output_res).select()
            return res

        if self.method == "ES":
            res = nimo.ai_tools.ai_tool_es.ES(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals).select()
            return res

        if self.method == "COMBI":
            res = nimo.ai_tools.ai_tool_combi.COMBI(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.physbo_score, self.minimization, self.combi_ranges, 
            self.spread_elements).select()
            return res


class preparation_input():
    """Class of preparation input

    This class can create input for robot experiments and star robot experiments.

    """

    def __init__(self, machine, input_file, input_folder):
        """Constructor
        
        This function do not depend on robot.

        Args:
            machine (str): "STAN" or "NAREE"
            input_file (str): the file for proposals from MI algorithm
            inputFolder (str): the folder where input files for robot are stored

        """

        self.machine = machine
        self.input_file = input_file
        self.input_folder = input_folder

        res = self.module_selection()



    def module_selection(self):
        """module selection of preparation input
        
        This function do not depend on robot.

        Returns:
            res (str): True for success, False otherwise.
        """

        if self.machine == "STAN":
            res = nimo.input_tools.preparation_input_standard.Standard(self.input_file, self.input_folder).perform()
            return res

        if self.machine == "NAREE":
            res = nimo.input_tools.preparation_input_naree.NAREE(self.input_file, self.input_folder).perform()
            return res

        if self.machine == "COMBAT":
            res = nimo.input_tools.preparation_input_combat.COMBAT(self.input_file, self.input_folder).perform()
            return res


class analysis_output():
    """Class of analysis output

    This class can analyze output.

    """

    def __init__(self, machine, input_file, output_file, num_objectives, output_folder, objectives_info = None):
        """Constructor
        
        This function do not depend on robot.

        Args:
            machine (str): "STAN" or "NAREE"
            input_file (str): the file for proposals from MI algorithm
            output_file (str): the file for candidates which will be updated in this script
            num_objectives (int): the number of objectives
            output_folder (str): the folder where the output files are stored by robot
            objectives_select (dict): the dictionary for objectives selection

        """

        self.machine = machine
        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.output_folder = output_folder
        self.objectives_info = objectives_info

        res = self.module_selection()


    def module_selection(self):
        """module selection of analysis input
        
        This function do not depend on robot.

        Returns:
            res (str): True for success, False otherwise.


        """

        if self.machine == "STAN":
            res = nimo.output_tools.analysis_output_standard.Standard(self.input_file, self.output_file, self.num_objectives, self.output_folder).perform()
            return res

        if self.machine == "NAREE":
            res = nimo.output_tools.analysis_output_naree.NAREE(self.input_file, self.output_file, self.num_objectives, self.output_folder, self.objectives_info).perform()
            return res

        if self.machine == "COMBAT":
            res = nimo.output_tools.analysis_output_combat.COMBAT(self.input_file, self.output_file, self.num_objectives, self.output_folder).perform()
            return res



class output_update():
    """Class of output update

    This class can update output to candidate file.

    """

    def __init__(self, input_file, output_file, num_objectives, objective_values):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for proposals from MI algorithm
            output_file (str): the file for candidates which will be updated in this script
            num_objectives (int): the number of objectives
            objective_values (list[float]): the list having objective function values

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.objective_values = objective_values

        res = nimo.output_tools.analysis_output_update.Update(input_file = self.input_file, 
                                    output_file = self.output_file, 
                                    num_objectives = self.num_objectives, 
                                    objective_values = self.objective_values).perform()



def history(input_file, num_objectives, itt = None, history_file = None):
    """Containing history results

    This function do not depend on robot.

    Args:
        input_file (str): the file for candidates
        num_objectives (int): the number of objectives
        itt (int): the number of step
        history_file (list[float]): the file for history results

    Returns:
        history_file (list[float]): the file for history results (updated)

    """

    if history_file is None:

        arr = np.genfromtxt(input_file, skip_header=1, delimiter=',')
        arr_train = arr[~np.isnan(arr[:, - 1]), :]

        X_train = arr_train[:, : - num_objectives].tolist()
        t_train = arr_train[:, - num_objectives:].tolist()

        history_file = []

        if len(X_train) != 0:

            for i in range(len(X_train)):
                history_file.append([0, X_train[i], t_train[i]])

    else:

        obs_X = []

        for i in range(len(history_file)):
            obs_X.append(history_file[i][1])

        arr = np.genfromtxt(input_file, skip_header=1, delimiter=',')
        arr_train = arr[~np.isnan(arr[:, - 1]), :]

        X_train = arr_train[:, : - num_objectives].tolist()
        t_train = arr_train[:, - num_objectives:].tolist()

        for i in range(len(X_train)):

            if X_train[i] not in obs_X:
                history_file.append([itt+1, X_train[i], t_train[i]])

    return history_file



class analysis():
    """Class of analysis

    This class can perform analyses.

    """

    def __init__(self, method, input_file, output_file = None, num_objectives = None, 
                 num_discretize = None, y_plot_range = None):
        """Constructor
        
        This function do not depend on robot.

        Args:
            method (str): "RE" or "BO"or "BLOX" or "PDC"
            input_file (str): the file for candidates for AI algorithm
            output_file (str): the file for proposals from AI algorithm
            num_objectives (int): the number of objectives

        """

        self.method = method
        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives

        self.num_discretize = num_discretize
        self.y_plot_range = y_plot_range
        
        res = self.module_selection()
        


    def module_selection(self):
        """module selection of preparation input
        
        This function do not depend on robot.

        Returns:
            res (str): True for success, False otherwise.

        """
        res = 'False'
        if self.method == "WAM":
            res = nimo.ai_tools.ai_tool_slesa_WAM.SLESA_WAM(self.input_file, 
            self.num_discretize, self.y_plot_range).calculation()
            return res


