import csv
import sys

import os


class Insert():
    """Class of Standard

    This class can perform analysis of outputs from robot.

    """

    def __init__(self, input_file, output_file, num_objectives, ndigits):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for parameters and objective function values
            output_file (str): the file for candidates which will be updated in this script
            num_objectives (int): the number of objectives
            ndigits (int): the number of decimal places for round up.

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.ndigits = ndigits

        if self.ndigits == None:
            self.ndigits = 5

    def perform(self):
        """perfroming analysis of output from robots

        This function do not depend on robot.
    
        Returns:
            res (str): True for success, False otherwise.

        """

        print("Start inserting objective function values into the candidates file!")
        
        res, p_List = self.load_data(self.input_file)

        if res == False:
            print("ErrorCode: error in load_data function")
            sys.exit()

        res = self.update_candidate_file(self.num_objectives, self.output_file, p_List)

        if res == False:
            print("ErrorCode: error in update_candidate_file function")
            sys.exit()


        print("Finish inserting objective function values into the candidates file!")

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

            res = True

        except:
            res = False

        return res, p_List


    def update_candidate_file(self, num_objectives, output_file, p_List):
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


            for iii in range(len(p_List) - 1):

                num = 0
                target_p = p_List[iii + 1][0 : - num_objectives]
                target_p = [round(float(target_p[i]), self.ndigits) for i in range(len(target_p))]

                for ii in range(len(c_List) - 1):

                    target_c = c_List[ii + 1][0 : - num_objectives]
                    target_c = [round(float(target_c[i]), self.ndigits) for i in range(len(target_c))]
     
                    if target_p == target_c:
                        c_List[ii + 1] = p_List[iii + 1]
                        num = 1

                if num == 0:
                    print(str(p_List[iii + 1][0 : - num_objectives]) + " is not found in the candidates file.")


            with open(output_file, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(c_List)

            res = True

        except:
            res = False

        return res

