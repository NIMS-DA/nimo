import csv
import sys
import time
import pathlib


class Standard():
    """Class of Standard

    This class can create input file for robot experiments and start the robot experiments.

    """

    def __init__(self, input_file, input_folder):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for proposals from MI algorithm
            input_folder (str): the folder where input files for robot are stored

        """

        self.input_file = input_file
        self.inputFolder = input_folder



    def perform(self):
        """perfroming preparation input and starting robot experiments 

        This function do not depend on robot.
    
        Returns:
            res (str): True for success, False otherwise.

        """

        print("Start preparation input!")

        res, p_List = self.load_data(self.input_file)

        if res == False:
            print("ErrorCode: error in load_data function")
            sys.exit()

        res = self.make_machine_file(p_List,self.inputFolder)

        if res == False:
            print("ErrorCode: error in make_machine_file function")
            sys.exit()

        res = self.send_message_machine(self.inputFolder)

        if res == False:
            print("ErrorCode: error in send_message_machine function")
            sys.exit()


        print("Finish preparation input!")

        return "True"



    def load_data(self, input_file):
        """Loading proposals

        This function do not depend on robot.
    
        Args:
            input_file (str): the file for proposals from AI algorithm

        Returns:
            res (bool): True for success, False otherwise.
            p_List (list[float]): the list of proposals

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



    def make_machine_file(self, p_List, inputFolder):
        """Making input files for robot

        This function DEPEND on robot.

        Args:
            p_List (list[float]): the list of proposals 
            inputFolder (str): the folder where the input files for robot are stored

        Returns:
            res (bool): True for success, False otherwise.

        """

        res = False

        try:
            dt_now = time.localtime()
            filepath = inputFolder + "/"  + time.strftime('%y%m%d%H%M%S', dt_now) + ".csv"

            with open(filepath, 'w') as f:
                f.write("input file for machine")
                f.write("\n")
                f.write(",".join(p_List[1][1:]))

            res = True  

        except:
            res = False

        return res


    def send_message_machine(self,inputFolder):
        """Sending a message to start the robot

        This function DEPEND on robot.

        Args:
            inputFolder (str): the folder where the input files for robot are stored

        Returns:
            res (bool): True for success, False otherwise.

        """

        res = False

        try:
            filepath = inputFolder+"/inputend.txt"

            touch_file = pathlib.Path(filepath)
            touch_file.touch()

            res = True
        
        except:
            res = False

        return res



