import csv
import sys

import os


class Update():
    """Class of Standard

    This class can perform analysis of outputs from robot.

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
        self.objs = objective_values


        if isinstance(objective_values[0],list) == False:

            self.objs = []

            for i in range(len(objective_values)):
                self.objs.append([objective_values[i]])



    def perform(self):
        """perfroming analysis of output from robots

        This function do not depend on robot.
    
        Returns:
            res (str): True for success, False otherwise.

        """

        print("Start update candidates file!")
        
        res, p_List, o_List = self.load_data(self.input_file)

        if res == False:
            print("ErrorCode: error in load_data function")
            sys.exit()

        res = self.update_candidate_file(self.num_objectives, self.output_file, o_List)

        if res == False:
            print("ErrorCode: error in update_candidate_file function")
            sys.exit()


        print("Finish update candidates file!")

        return "True"



    def load_data(self, input_file):
        """Loading proposals

        This function do not depend on robot.

        Args:
            input_file (str): the file for proposals from MI algorithm

        Returns:
            res (bool): True for success, False otherwise.
            p_List (list[float]): list of proposals

        """

        p_List = []
        o_List = []

        try:
            with open(input_file) as inf:
                reader = csv.reader(inf)
                p_List = [row for row in reader]

            for i in range(len(p_List)-1):
                o_List.append([p_List[i+1][0],self.objs[i]])


            res = True

        except:
            res = False

        return res, p_List, o_List


    def update_candidate_file(self, num_objectives, output_file, o_List):
        """Updating candidates

        This function do not depend on robot.

        Args:
            num_objectives (int): the number of objectives
            output_file (str): the file for candidates
            o_List (list[float]): the list of objectives

        Returns:
            res (bool): True for success, False otherwise.

        """


        try:
            with open(output_file) as inf:
                reader = csv.reader(inf)
                c_List = [row for row in reader]

            for i in range(len(o_List)):
                combi_list = c_List[int(o_List[i][0]) + 1][0 : - num_objectives] + o_List[i][1]
                c_List[int(o_List[i][0]) + 1] = combi_list

            with open(output_file, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(c_List)

            res = True

        except:
            res = False

        return res

