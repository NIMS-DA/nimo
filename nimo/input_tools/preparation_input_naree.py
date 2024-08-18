import csv
import sys
import time
import pathlib
import os


class NAREE():
    """Class of NAREE

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

        res = self.send_message_machine()

        if res == False:
            print("ErrorCode: error in send_message_machine function")
            sys.exit()

        print("Finish preparation input!")

        return "True"



    def load_data(self, input_file):
        """Loading proposals

        This function do not depend on robot.

        Args:
            input_file (str): the file for proposals from MI algorithm

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

        WELL_ROW = 96     ## PLATE WELL ROW Num
        WELL_COL = 96     ## PLATE WELL COLUMN Num

        res = True
        
        ## 96 * 96 のバッファを確保
        w_List = [["0" for x in range(WELL_COL)] for i in range(WELL_ROW)]	
        
        ## proposals.csv から要素数 N を取り出す
        idx = 0
        sampleNo = [0] * 96
            
        # CSV見出し行の母液WELL番号をsampleNoバッファへ取り出す(最大96個分)
        for i,sample in enumerate(p_List[0]):
            if i != 0:
                sampleNo[i-1] = int(sample)
            
        ## CSVの各行毎の要素（母液量）を抽出しwellbuff配列に展開する
        crow = 0
        for chbuff in p_List:
            i=0
            cclm = 0
            for well in chbuff:
                if i != 0:
                    if crow > 0:
                        if (well != "") and (well != "0") and (well != "0.0"):
                            w_List[crow-1][(sampleNo[cclm])-1] = well
                    cclm = cclm + 1
                i = i + 1
            crow = crow + 1

        dt_now = time.localtime()
        strdate = inputFolder + "\\" + time.strftime('%y%m%d%H%M%S', dt_now) + ".csv"  # YY/MM/DD hh:mm:ssに書式化
        platea = ["A","B","C","D","E","F","G","H"]
        with open(strdate, 'w', encoding='cp932', newline='\n') as f:
            wdata = ",,溶液①,,溶液②,,溶液③,,溶液④,,溶液⑤,,溶液⑥,,溶液⑦,,溶液⑧,,溶液⑨,,溶液⑩,\r\n"
            f.write(wdata)
            wdata = "well,ID,No,uL,No,uL,No,uL,No,uL,No,uL,No,uL,No,uL,No,uL,No,uL,No,uL\r\n"
            f.write(wdata)

            for i in range(WELL_ROW):
                wdata = platea[(i%8)] + "," + str(int(i / 8)+1)
                spcount = 10
                for j in range(WELL_COL):
                    if float(w_List[i][j]) != 0.0:
                        wdata = wdata + "," + str(j+1) + "," + w_List[i][j]
                        spcount = spcount - 1
                if spcount < 0:
                    print("ErrorCode:combination 10 Over")	## エラーなし
                    return
                if spcount > 0:
                    for k in range(spcount):
                        wdata = wdata + ",,"
                    wdata = wdata + "\r\n"
                f.write(wdata)
            f.close()
        

        ## 装置固有のCSVを正常にフォルダへ書き込んだ事を知らせる。空のメッセージファイルを書き込む
        strdate = inputFolder + "\\inputend.txt"

        touch_file = pathlib.Path(strdate)
        touch_file.touch()

        return res


    def send_message_machine(self):
        """Sending a message to start the robot

        This function DEPEND on robot.

        Returns:
            res (bool): True for success, False otherwise.

        """

        res = True
        
            ## 送信タイミングファイル
            ##feaeLib.udp_send()		## UDP 送信

            ## CSV File copy end check
        while True:
                ## 送信タイミングファイル
                ##feaeLib.udp_send()		## UDP 送信
        
                ## メッセージファイルが削除されていれば、装置が正常にＣＳＶを取り込んでくれた。
            ChkRes = self.fileendCheck(self.inputFolder)
            if ChkRes == True:
                break
            time.sleep(2)

        return res


    def fileendCheck(self, inputFolder):
        """Sending a message to start the robot

        This function DEPEND on robot.
        
        Args:
            inputFolder (str): the folder where the inputend file is stored

        Returns:
            res (bool): True for success, False otherwise.

        """

        if os.path.isfile(inputFolder+"\\inputend.txt") == False:    ## ファイル存在なし
            res = True
        else:
            res = False

        return res
