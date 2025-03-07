import csv
import sys
import time
import pathlib


class COMBAT():
    """Class of COMBAT

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

        #res = self.send_message_machine(self.inputFolder)

        #if res == False:
        #    print("ErrorCode: error in send_message_machine function")
        #    sys.exit()


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

            # elements = [Fe, Co, Ni, Ta, W, Ir]
            Density = [7.874, 8.9, 8.908, 16.654, 19.25, 22.562]
            Weight = [55.845, 58.933195, 58.6934, 180.94788, 183.84, 192.217]
            Power = [100, 100, 80, 10, 10, 10]
            Rate = [0.00813005, 0.0135258, 0.0118133, 0.00621467, 0.00584, 0.00492]


            target1 = [float(p_List[1][i + 1]) for i in range(len(p_List[1]) - 1)]

            target2 = [float(p_List[-1][i + 1]) for i in range(len(p_List[-1]) - 1)]

            inclination = []

            for ii in range(len(target1)):
                if target1[ii] - target2[ii] != 0:
                    inclination.append(1)
                else:
                    inclination.append(0)


            Numbers = len(target1) + 2

            thickness_ratio1 = []
            time1 = []

            for i in range(len(target1)):

                thickness_ratio1.append(Density[0] * Weight[i] * target1[i] / (Weight[0] * Density[i] * target1[0]))


            for i in range(len(target1)):

                time1.append(round(0.5 * thickness_ratio1[i] / sum(thickness_ratio1) / Rate[i]))


            thickness_ratio2 = []
            time2 = []

            for i in range(len(target2)):

                thickness_ratio2.append(Density[0] * Weight[i] * target2[i] / (Weight[0] * Density[i] * target2[0]))


            for i in range(len(target2)):

                time2.append(round(0.5 * thickness_ratio2[i] / sum(thickness_ratio2) / Rate[i]))



            string = []

            #homogeneous
            for ii in range(len(target1)):

                for jj in range(len(target1)):
                
                    if ii == jj:
                        string.append(str(Power[ii]))
                    else:
                        string.append("0")

                for jj in range(len(target1)):
                
                    if ii == jj:
                        string.append(str(min([time1[ii],time2[ii]])))
                    else:
                        string.append("0")
                    
                string = string + ["-11", "-11", "0", "0"]

                string.append(r"0\0D\0A")

            #inclination

            incli_num = 0

            for ii in range(len(target1)):

                if inclination[ii] == 1:
                
                    for jj in range(len(target1)):
                
                        if ii == jj:
                            string.append(str(Power[ii]))
                        else:
                            string.append("0")
                
                    for jj in range(len(target1)):
                
                        if ii == jj:
                            string.append(str(abs(time1[ii] - time2[ii])))
                        else:
                            string.append("0")

                    if incli_num == 0:
                        string = string + ["-3", "3", "0", "0"]
                        string.append(r"0\0D\0A")
                    if incli_num == 1:
                        string = string + ["-3", "3", "180", "180"]
                        string.append(r"1")

                    incli_num += 1

            result = ','.join(string)

            result = result.replace('0A,', '0A')

            print("experimental settings")
            print(result)


            datalist = ['[recipe]\n', 'cycle = 60    \n', 'note = "FeCoNiTaWIr"\n', 'data = \n', '\n', '[setup]\n', 'subtemp = 0    \n', 'mfc1 = 50    \n', 'mfc2 = 0    \n', 'mfc3 = 0    \n', 'cv = 0.800000    \n', 'pre_sput = 1    \n', 'pause = FALSE    \n', 'continuous = FALSE    \n', 'material = "Fe,Co,Ni,Ta,W,Ir\\0D\\0A"\n']

            for ii in range(len(datalist)):

                if datalist[ii] == 'data = \n':
                    datalist[ii] = 'data = \"' + result + "\"\n"

            f = open('new_recipe.rcp', 'w')

            f.writelines(datalist)

            f.close()


            min_thickness = [ min([time1[i], time2[i]]) * Rate[i] for i in range(6) ]
            max_thickness = [ max([time1[i], time2[i]]) * Rate[i] for i in range(6) ]


            atom_percent = [p_List[0]]

            for ii in range(13):

                each_thickness = []

                already = 0

                for jj in range(len(inclination)):

                    if inclination[jj] == 0:
                        
                        each_thickness.append(round(min_thickness[jj], 6))

                    elif inclination[jj] == 1 and already == 0:

                        each_thickness.append(round(min_thickness[jj] + (max_thickness[jj] - min_thickness[jj]) / 6 * 0.5 * ii, 6))

                        already = 1
                    
                    elif inclination[jj] == 1 and already == 1:
                        
                        each_thickness.append(round(max_thickness[jj] - (max_thickness[jj] - min_thickness[jj]) / 6 * 0.5 * ii, 6))

                        already = 2

                each_atom_num = []

                for jj in range(len(each_thickness)):

                    each_atom_num.append(round(each_thickness[jj] * Density[jj] / Weight[jj], 6))


                each_atom_percent = [-1]

                for jj in range(len(each_atom_num)):

                    each_atom_percent.append(round(each_atom_num[jj] * 100 / sum(each_atom_num), 1))


                atom_percent.append(each_atom_percent)



            with open("proposals_real.csv", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(atom_percent)

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



