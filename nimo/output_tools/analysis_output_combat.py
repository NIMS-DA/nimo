import csv
import time
import sys
import os

import numpy as np


class COMBAT():
    """Class of COMBAT

    This class can perform analysis of outputs from robot.

    """

    def __init__(self, input_file, output_file, num_objectives, output_folder):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for proposals from MI algorithm
            output_file (str): the file for candidates which will be updated in this script
            num_objectives (int): the number of objectives
            output_folder (str): the folder where the output files are stored by robot

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.output_folder = output_folder



    def perform(self):
        """perfroming analysis of output from robots

        This function do not depend on robot.
    
        Returns:
            res (str): True for success, False otherwise.

        """

        print("Start analysis output!")

        #res = self.recieve_exit_message(self.output_folder)

        #if res == False:
        #    print("ErrorCode: error in recieve_exit_message function")
        #    sys.exit()
        
        res, p_List = self.load_data(self.input_file)

        if res == False:
            print("ErrorCode: error in load_data function")
            sys.exit()

        res, o_List = self.extract_objectives(self.num_objectives, self.output_folder, p_List)

        if res == False:
            print("ErrorCode: error in extract_objectives function")
            sys.exit()

        res = self.update_candidate_file(self.num_objectives, self.output_file, p_List, o_List)

        if res == False:
            print("ErrorCode: error in update_candidate_file function")
            sys.exit()


        print("Finish analysis output!")

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

        try:
            with open(input_file) as inf:
                reader = csv.reader(inf)
                p_List = [row for row in reader]
                
            res = True

        except:
            res = False

        return res, p_List 



    def recieve_exit_message(self,output_folder):
        """Recieving exit message from machine

        This function DEPENDS on robot.

        Args:
            output_folder (str): the folder where the results by machine are stored

        Returns:
            res (bool): True for success, False otherwise.

        """

        try:
            filepath = output_folder + "/outputend.txt"

            while not(os.path.isfile(filepath)):
                time.sleep(10)

            os.remove(filepath)

            res = True


        except:
            res = False

        return res



    def extract_objectives(self, num_objectives, output_folder, p_List):    
        """Extracting objective values from output files by robot

        This function DEPENDS on robot.

        Args:
            num_objectives (int): the number of objectives
            output_folder (str): the folder where the results by machine are stored
            p_List (list[float]): the list of proposals

        Returns:
            res (bool): True for success, False otherwise.
            o_List (list[float]): the list of objectives

        """

        from sklearn.metrics import mean_squared_error

        o_List = []

        try:
            filepath = output_folder + "/exp_results.csv"

            ###########################
            # Hall voltage calculation
            ###########################
            dataset = []

            with open(filepath) as f:
                for line in f:
              
                   dataset.append(line.strip('\n').split("\t"))

            del dataset[0]


            Hall_voltage = []


            for ii in range(len(dataset[0]) - 2):

                x1 = []
                y1 = []
                x2 = []
                y2 = []

                for kk in range(len(dataset)):

                    if float(dataset[kk][1]) > 25000:
                        x1.append(float(dataset[kk][1]))  
                        y1.append(float(dataset[kk][ii + 2])) 

                    if float(dataset[kk][1]) < - 25000:
                        x2.append(float(dataset[kk][1]))  
                        y2.append(float(dataset[kk][ii + 2])) 
                

                a1, b1 = np.polyfit(x1, y1, 1)

                #fitting function
                f1 = a1 * np.array(x1) + b1

                rmse1 = np.sqrt(mean_squared_error(f1, y1))

                
                a2, b2 = np.polyfit(x2, y2, 1)

                #fitting function
                f2 = a2 * np.array(x2) + b2

                rmse2 = np.sqrt(mean_squared_error(f2, y2))


                #print("ch", ii + 1, b1, rmse1, b2, rmse2, (b1 - b2) / 2 / 0.0002 * 30 * 0.1)


                min_y = np.mean(y2)
                max_y = np.mean(y1)

                def_y = max_y - min_y
                    
                count = 0
                
                for kk in range(len(dataset)):

                    if float(dataset[kk][ii + 2]) > max_y + def_y * 0.1:
                        count += 1

                    if float(dataset[kk][ii + 2]) < min_y - def_y * 0.1:
                        count += 1

                if rmse1 > 10 ** -5 or rmse2 > 10 ** -5:
                    Hall_voltage.append(0.0)
                
                elif count > 5:
                    Hall_voltage.append("")

                else:
                    val_Hall_voltage = (b1 - b2) / 2 / 0.0002 * 30 * 0.1

                    if val_Hall_voltage < 0:
                        Hall_voltage.append(0)
                    else:
                        Hall_voltage.append(val_Hall_voltage)


            o_List = [[Hall_voltage[i]] for i in range(len(Hall_voltage))]

            res = True

        except:
            res = False

        return res, o_List


    def update_candidate_file(self, num_objectives, output_file, p_List, o_List):
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

            proposals = []

            for ii in range(len(p_List) - 1):

                row = [float(p_List[ii + 1][i + 1]) for i in range(len(p_List[0]) - 1)]
 
                row.append(o_List[ii][0])

                proposals.append(row)

            minmax = []

            for ii in range(len(proposals[0]) - 1):

                minmax.append([np.min([r[ii] for r in proposals]), np.max([r[ii] for r in proposals])])



            new_dataset = []


            with open(self.output_file) as f:
                reader = csv.reader(f)
                header = next(reader)

                new_dataset.append(header)

                for row in reader:

                    if_index = 0

                    for ii in range(len(minmax)):

                        if float(row[ii]) >= minmax[ii][0] and float(row[ii]) <= minmax[ii][1]:

                            if_index += 1

                    if if_index != len(minmax):
                        new_dataset.append(row)



            for ii in range(len(proposals)):

                new_dataset.append(proposals[ii])


            with open('candidates.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerows(new_dataset)


            res = True

        except:
            res = False

        return res

