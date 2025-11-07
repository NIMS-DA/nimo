import csv
import sys
import time
import pathlib
import os

CERTUS_Mapping_File = './CERTUS_Mapping_Table.csv'
class CERTUS_SC():
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
        
        chunk_size = 36
        num_certus_bottles = 8
        
        def get_process_count(n, chunk_size):
            return max(1, (n + chunk_size - 1) // chunk_size)

        
        def make_well_list():
            well_list = []
            for cha in range(8):
                n = 0
                if cha % 2 == 0 :
                    n = 1
                else:
                    n = 2
                while n < 10:
                    well_list.append(chr(cha + 65)+str(n))
                    n += 2
            return well_list

        
        def write_dispense_file(path,data):
            with open(path,'w',newline = '') as f:
                writer = csv.writer(f,delimiter=';')
                writer.writerows(data)
        
        
        res = False
        dt_now = time.localtime()
        dummy,mapping_table = self.load_data(CERTUS_Mapping_File)
        CERTUS_list = []
        well_list = make_well_list()
        hedder = ['Position'] + mapping_table[0][len(p_List[0]):len(p_List[0]) + num_certus_bottles]
        
        for row in p_List:
            for table_row in mapping_table:
                if row[0] == table_row[0]:
                    CERTUS_list.append(['{:.2f}'.format(float(r)) for r in table_row[len(p_List[0]):]])
        
        for filecount in range(0,(get_process_count(len(CERTUS_list),chunk_size))):
            proposal_list =  [row for row in CERTUS_list[0+filecount*chunk_size:chunk_size+filecount*chunk_size]]

            tmp_data = []
            for i in range(0,(len(proposal_list[0]) // num_certus_bottles)):
                tmp_data.append([row[0+i*num_certus_bottles:num_certus_bottles+i*num_certus_bottles] for row in proposal_list])
            if len(proposal_list[0]) % num_certus_bottles != 0 :
                tmp_data.append([row[-(len(proposal_list[0]) % num_certus_bottles):]+['0.00' for _ in range(num_certus_bottles-len(proposal_list[0]) % num_certus_bottles)] for row in proposal_list])
    
            for chr_count,r in enumerate(tmp_data):
                filedata = []
                filedata.append(hedder)
                for well_count, n in enumerate(r):
                    filedata.append([well_list[well_count]]+n)
                if len(proposal_list[0]) % num_certus_bottles != 0 :
                    suffix = '_' + str(filecount) + '_' + chr(chr_count + 65) + '.csv'
                else:
                    suffix = '_' + str(filecount) + '.csv'
                write_dispense_file(os.path.join(inputFolder,(time.strftime('%y%m%d%H%M%S', dt_now) + suffix)), filedata)
            
        """
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
        """
        res = True
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



