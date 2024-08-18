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
                 physbo_score = None,
                 ard = None,
                 output_res = None):

        """Constructor
        
        This function do not depend on robot.

        Args:
            method (str): "RE" or "BO"or "BLOX" or "PDC"
            input_file (str): the file for candidates for AI algorithm
            output_file (str): the file for proposals from AI algorithm
            num_objectives (int): the number of objectives
            num_proposals (int): the number of proposals

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
        self.ard = ard

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
            self.num_objectives, self.num_proposals, self.re_seed).select()
            return res

        if self.method == "PHYSBO":
            res = nimo.ai_tools.ai_tool_physbo.PHYSBO(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals, self.physbo_score, self.ard, self.output_res).select()
            return res

        if self.method == "PDC":
            res = nimo.ai_tools.ai_tool_pdc.PDC(self.input_file, self.output_file, 
            self.num_objectives, self.num_proposals).select()
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
            self.re_seed).select()
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


