import csv
import time
import sys
import numpy as np
import os
import datetime
import socket
import math

class NAREE():
    """Class of NAREE

    This class can create input file for robot experiments and start the robot experiments.

    """

    def __init__(self, input_file, output_file, num_objectives, output_folder, objectives_info):
        """Constructor

        This function do not depend on robot.

        Args:
            input_file (str): the file for proposals from MI algorithm
            output_file (str): the file for candidates which will be updated in this script
            num_objectives (int): the number of objectives
            output_folder (str): the folder where the output files are stored by robot
            objectives_info (dict): the dictionalry objectives infomation (オブジェクト選択番号　＆　乗数）

        """

        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.output_folder = output_folder
        self.object_info = objectives_info

    def perform(self):
        """perfroming analysis of output from robots

        This function do not depend on robot.

        Returns:
            res (str): True for success, False otherwise.

        """


        print("Start analysis output!")

        res = self.recieve_exit_message()

        if res == False:
            print("ErrorCode: error in recieve_exit_message function")
            sys.exit()

        res, p_List = self.load_data(self.input_file)

        if res == False:
            print("ErrorCode: error in load_data function")
            sys.exit()

        res = self.objectives_create_SD8(self.num_objectives, self.output_folder, self.output_folder,p_List)
        if res == False:
            print("ErrorCode: error in extract_objectives function")
            sys.exit()

        res, o_List = self.extract_objectives(self.num_objectives, self.output_folder, p_List)

        if res == False:
            print("ErrorCode: error in extract_objectives function")
            sys.exit()

        res = self.update_candidate_file(self.num_objectives, self.output_file, o_List)

        if res == False:
            print("ErrorCode: error in update_candidate_file function")
            sys.exit()

        print("Finish analysis output!")

        return "True"



    def load_data(self,input_file):
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



    def recieve_exit_message(self):
        """Recieving exit message from machine

        This function DEPENDS on robot.

        Returns:
            res (bool): True for success, False otherwise.

        """

        try:
            '''
             ## Mesure File copy end check
            res = self.udp_Receive()  ## UDP で測定ファイル完了をチェック（メッセージが届くまで待つ）
            if res == False:  ## UDP 受信エラー
                print("updReceive Error")
                return False
           '''

            filepath = self.output_folder + "\\outputend.txt"

            while not(os.path.isfile(filepath)):
                time.sleep(10)

            time.sleep(1)
            os.remove(filepath)

            res = True

        except:
            res = False

        return res

    def objectives_create_SD8(self,num_objectives, result_folder, output_folder,p_List ) :
        """

        Args:
            num_objectives (int):  the number of objectives
            result_folder (str):  the folder where the results of measurement SD8 stored
            output_folder (str): the folder where the results by machine are stored
            p_List (list[float]): the list of proposals


        Returns:
            res (bool): True for success, False otherwise.

        """

        objData = []
        objData.clear()

        try:
            filepath = output_folder + "\\measfolder.txt"
            with open(filepath) as inf:
                measfolder = inf.read()     # SD8 フォルダ読み込み
                measfolder = measfolder.replace('\r', "")
                measfolder = measfolder.replace('\n', "")
                self.measFolder = measfolder

            filepath = result_folder + "\\results.csv"
            with open(filepath,"w",newline="") as wf:

                for i in range(len(p_List) - 1):
                    wdata = list('SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS')
                    res,objData = self.objects_retrieval((i+1),measfolder)
                    if len(objData) != 0:
                        for j,data in enumerate(objData):
                            wdata[j] = data

                    writer = csv.writer(wf)
                    writer.writerow(wdata)

            res = True
        except:
            res = False
        return res

    def objects_retrieval(self,chno,measPath):
        """

        Args:
            chno (int): Measurement channel number (1 to 96)
            measPath (str): Folder containing measurement results

        Returns:
            res (bool): True for success, False otherwise.
            objData(list): Measurement results

        """
        res = False

        objData = []

        objData.clear()

        finalVoltage = "0"   # 最終電圧値
        onecyclesteptime = ""   # 最終Step time
        measFileName = measPath + "\\Ch_" + str(chno) + "\\Ch_" + str(chno) + "-001.csv"
        print(measFileName)

        ##  ファイル有無チェック
        if os.path.isfile(measFileName) == False:  ## ファイル存在しない
            print("------ Error! no Measurement File !")
            return res,objData

        with open(measFileName, 'r') as input_data:

            V_List = []  ## Voltage data
            S_List = []  ## Step time data
            M_List = []  ## Mode data

            try:

                for row in csv.reader(input_data):
                    V_List.append(row[1])   ## Voltage Column
                    S_List.append(row[8])   ## Step time Column
                    M_List.append(row[11])  ## MODE Column

                del V_List[0]
                del V_List[0]
                del S_List[0]
                del S_List[0]
                del M_List[0]
                del M_List[0]

                ## 最終電圧値セット
                finalVoltage = row[1]
                objData.append(finalVoltage)
                ## １サイクル終了ステップ時間セット
                if ('Discharge' == row[11]): 
                    onecyclesteptime = row[8]
                    objData.append(onecyclesteptime)


                data = np.zeros(10, dtype={'names': ('DischargeST', 'DischargeED', 'ChargeST', 'ChargeED'), 'formats': (
                'i4', 'i4', 'i4', 'i4')})  # U10 -> Unicode(10文字まで), i4 -> int32, f8 -> float64

                ## coulomb data(Step値が2400未満NG、Step時間が0ならNG,空白ならNG)
                cou_data = np.zeros(10, dtype={'names': ('DisStep', 'ChaStep', 'CE'), 'formats': (
                'f4', 'f4', 'f4')})  # U10 -> Unicode(10文字まで), i4 -> int32, f4 -> float32,  f8 -> float64

                ## over potential data
                op_data_step = np.zeros(10, dtype={
                    'names': ('Cha1_2mv', 'Cha1_3mv', 'Cha2_3mv', 'Dis1_2mv', 'Dis1_3mv', 'Dis2_3mv'), 'formats': (
                    'i4', 'i4', 'i4', 'i4', 'i4', 'i4')})  # U10 -> Unicode(10文字まで), i4 -> int32, f4 -> float32,  f8 -> float64
                op_data = np.zeros(10, dtype={'names': (
                'Cha1_2mv', 'Cha1_3mv', 'Cha2_3mv', 'Dis1_2mv', 'Dis1_3mv', 'Dis2_3mv', 'OP1_2mv', 'OP1_3mv', 'OP2_3mv'),
                                              'formats': ('f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4',
                                                          'f4')})  # U10 -> Unicode(10文字まで), i4 -> int32, f4 -> float32,  f8 -> float64

                ## over potential Cycle DiFF(100以下ならOK)
                op_diff_data = np.zeros(10, dtype={'names': ('OP_Diff1_2', 'OP_Diff1_3', 'OP_Diff2_3'),
                                                   'formats': ('f4', 'f4', 'f4')})

                ## Max,Min data(200未満ならOK)
                maxmin_data = np.zeros(10, dtype={'names': ('ChaMin', 'ChaMax', 'DisMin', 'DisMax', 'ChaDiff', 'DisDiff'),
                                                  'formats': ('f4', 'f4', 'f4', 'f4', 'f4',
                                                              'f4')})  # U10 -> Unicode(10文字まで), i4 -> int32, f4 -> float32,  f8 -> float64

                r_to_dst = ["Rest", "Discharge"]
                r_to_ded = ["Discharge", "Rest"]
                r_to_cst = ["Rest", "Charge"]
                r_to_ced = ["Charge", "Rest"]

                hitcell = list(range(0))

                backdata = ""
                status = 0
                index = 0
                rowNo = 1

                stepsetflag = False

                for row in M_List:
                    if (backdata == r_to_dst[0]) & (row == r_to_dst[1]):  # Rest → Discargeの切り替わり？
                        data[index]['DischargeST'] = rowNo + 1  ## Discharge開始行セット
                        status = 1
                        hitcell.append(rowNo)
                    # print("--->Discharge Start(" + str(rng1) + ") ---row="+str(rowNo))
                    elif (backdata == r_to_ded[0]) & (row == r_to_ded[1]):  # Discharge →　Restの切り替わり？
                        if stepsetflag == False:
                            stepsetflag = True
                            onecyclesteptime = S_List[rowNo-2]  ## １サイクル終了ステップ時間セット
                            objData.append(onecyclesteptime)
                        if status == 1:
                            data[index]['DischargeED'] = rowNo  ## Discharge終了行セット
                            status = 2
                        hitcell.append(rowNo)
                    # print('--->Discharge End(' + str(rng1)+ ") ---row="+str(rowNo))
                    elif (backdata == r_to_cst[0]) & (row == r_to_cst[1]):  # Rest → Chargeの切り替わり？
                        if status == 2:
                            data[index]['ChargeST'] = rowNo + 1  ## Charge開始行セット
                            status = 3
                        hitcell.append(rowNo)
                    # print('--->Charge Start(' + str(rng1)+ ") ---row="+str(rowNo))
                    elif (backdata == r_to_ced[0]) & (row == r_to_ced[1]):  # Charge →　Restの切り替わり？
                        if status == 3:
                            data[index]['ChargeED'] = rowNo  ## Charge終了行セット
                            status = 0
                            # 　Rest → Diacharge →　Rset →　Charge →　Rest　ならば　測定サイクル数更新
                            index = index + 1
                        hitcell.append(rowNo)
                    backdata = row
                    rowNo = rowNo + 1

            except:
                print("------ Error! Cant get data !")
                if onecyclesteptime == "":
                    onecyclesteptime = 'S'
                    objData.append(onecyclesteptime)
                return res,objData

        ## 最終行がChargeで終わっていれば、最終ステップとして登録する。
        if (status == 3) and (index == 2):
            data[index]['ChargeED'] = rowNo  ## Charge終了行セット
            status = 0
            index = index + 1  ## 測定サイクル数更新
            hitcell.append(rowNo - 2)
            # print('--->Charge End(' + str(rng1)+ ") ---row="+str(rowNo))
            ## クーロン効率値算出
            try:
                for i in range(index):
                    # Diacharge / Charge 終了行に対応したStep時間カラムの値からクーロン効率値を求める
                    cou_data[i]['DisStep'] = float(S_List[(data[i]['DischargeED'] - 2)])
                    cou_data[i]['ChaStep'] = float(S_List[(data[i]['ChargeED'] - 2)])
                    cou_data[i]['CE'] = float(cou_data[i]['ChaStep'] / cou_data[i]['DisStep'])

                    if (data[i]['DischargeED'] - data[i]['DischargeST']) < 2400:
                        ## ステップ時間2400未満エラー
                        print("------ Error! step time 2400 less than 2400 hours --ED= " + str(
                            data[i]['DischargeED']) + " --ST=" + str(data[i]['DischargeST']))
                        return res,objData

                    if (data[i]['DischargeST'] == 0) or (data[i]['DischargeED'] == 0) or (data[i]['ChargeST'] == 0) or (
                            data[i]['ChargeED'] == 0):
                        ## ステップ時間 0エラー
                        print("------ Error! step time zero")
                        return res,objData

                    ## Chrge 1/2,1/3,2/3,1/5,4/5 Row番号
                Chrow12 = []
                Chrow13 = []
                Chrow23 = []
                Chrow15 = []
                Chrow45 = []
                ## Dis 1/2,1/3,2/3,1/5,4/5 Row番号
                Disrow12 = []
                Disrow13 = []
                Disrow23 = []
                Disrow15 = []
                Disrow45 = []

                for i in range(index):
                    Chrow12.append(int((data[i]['ChargeED'] - data[i]['ChargeST']) / 2) + data[i]['ChargeST'])
                    Chrow13.append(int((data[i]['ChargeED'] - data[i]['ChargeST']) / 3) + data[i]['ChargeST'])
                    Chrow23.append(int((data[i]['ChargeED'] - data[i]['ChargeST']) / 3 * 2) + data[i]['ChargeST'])
                    Chrow15.append(int((data[i]['ChargeED'] - data[i]['ChargeST']) / 5) + data[i]['ChargeST'])
                    Chrow45.append(int((data[i]['ChargeED'] - data[i]['ChargeST']) / 5 * 4) + data[i]['ChargeST'])

                    Disrow12.append(int((data[i]['ChargeED'] - data[i]['ChargeST']) / 2) + data[i]['DischargeST'])
                    Disrow13.append(int((data[i]['ChargeED'] - data[i]['ChargeST']) / 3) + data[i]['DischargeST'])
                    Disrow23.append(int((data[i]['ChargeED'] - data[i]['ChargeST']) / 3 * 2) + data[i]['DischargeST'])
                    Disrow15.append(int((data[i]['ChargeED'] - data[i]['ChargeST']) / 5) + data[i]['DischargeST'])
                    Disrow45.append(int((data[i]['ChargeED'] - data[i]['ChargeST']) / 5 * 4) + data[i]['DischargeST'])

                ## over potential値算出
                for i in range(index):
                    # opd:np.float16
                    # opi:np.int32

                    # opi = Chrow12[i]
                    # opd = V_List[(Chrow12[i])]
                    # opd = float(V_List[(Chrow12[i])]) * 1000.0

                    op_data_step[i]['Cha1_2mv'] = Chrow12[i]  ## mv -> v  Charge 1/2 Chage Voltege
                    op_data_step[i]['Cha1_3mv'] = Chrow13[i]  ## mv -> v  Charge 1/3 Chage Voltege
                    op_data_step[i]['Cha2_3mv'] = Chrow23[i]  ## mv -> v  Charge 2/3 Chage Voltege

                    op_data_step[i]['Dis1_2mv'] = Disrow12[i]  ## mv -> v  DisCharge 1/2 DisChage Voltege
                    op_data_step[i]['Dis1_3mv'] = Disrow13[i]  ## mv -> v  DisCharge 1/3 DisChage Voltege
                    op_data_step[i]['Dis2_3mv'] = Disrow23[i]  ## mv -> v  DisCharge 2/3 DisChage Voltege

                    op_data[i]['Cha1_2mv'] = float(V_List[(Chrow12[i] - 2)]) * 1000.0  ## mv -> v  Charge 1/2 Chage Voltege
                    op_data[i]['Cha1_3mv'] = float(V_List[(Chrow13[i] - 2)]) * 1000.0  ## mv -> v  Charge 1/3 Chage Voltege
                    op_data[i]['Cha2_3mv'] = float(V_List[(Chrow23[i] - 2)]) * 1000.0  ## mv -> v  Charge 2/3 Chage Voltege

                    op_data[i]['Dis1_2mv'] = float(
                        V_List[(Disrow12[i] - 2)]) * 1000.0  ## mv -> v  DisCharge 1/2 DisChage Voltege
                    op_data[i]['Dis1_3mv'] = float(
                        V_List[(Disrow13[i] - 2)]) * 1000.0  ## mv -> v  DisCharge 1/3 DisChage Voltege
                    op_data[i]['Dis2_3mv'] = float(
                        V_List[(Disrow23[i] - 2)]) * 1000.0  ## mv -> v  DisCharge 2/3 DisChage Voltege

                    op_data[i]['OP1_2mv'] = (float(V_List[Chrow12[i] - 2]) - float(
                        V_List[Disrow12[i] - 2])) * 1000.0  ## mv -> v  1/2 over potential
                    op_data[i]['OP1_3mv'] = (float(V_List[Chrow13[i] - 2]) - float(
                        V_List[Disrow13[i] - 2])) * 1000.0  ## mv -> v  1/3 over potential
                    op_data[i]['OP2_3mv'] = (float(V_List[Chrow23[i] - 2]) - float(
                        V_List[Disrow23[i] - 2])) * 1000.0  ## mv -> v  2/3 over potential

            except:
                ## 計算エラー
                print("------ Error! Coulomb value or OP calculation error")
                if onecyclesteptime == "":
                    onecyclesteptime = 'S'
                    objData.append(onecyclesteptime)
                return res,objData

            ## over potential Cycle DiFF 算出(100以下ならOK)
            for i in range(index-1):
                op_diff_data[i]['OP_Diff1_2'] = abs(op_data[i + 1]['OP1_2mv'] - op_data[i]['OP1_2mv'])
                op_diff_data[i]['OP_Diff1_3'] = abs(op_data[i + 1]['OP1_3mv'] - op_data[i]['OP1_3mv'])
                op_diff_data[i]['OP_Diff2_3'] = abs(op_data[i + 1]['OP2_3mv'] - op_data[i]['OP2_3mv'])

            ## Chage Max,Min 算出（200未満ならOK）
            for i in range(index):
                lpdata = (int((data[i]['ChargeED'] - data[i]['ChargeST']) / 5 * 4)) - (
                    int((data[i]['ChargeED'] - data[i]['ChargeST']) / 5))
                stdt = int((data[i]['ChargeED'] - data[i]['ChargeST']) / 5)
                mindt = 10000.0
                for j in range(lpdata):
                    wdata = float(V_List[(data[i]['ChargeST'] + j + stdt) - 2]) * 1000.0
                    if wdata < mindt:
                        mindt = wdata
                maxdt = -10000
                for j in range(lpdata):
                    wdata = float(V_List[(data[i]['ChargeST'] + j + stdt) - 2]) * 1000.0
                    if wdata > maxdt:
                        maxdt = wdata

                maxmin_data[i]['ChaMin'] = mindt
                maxmin_data[i]['ChaMax'] = maxdt
                maxmin_data[i]['ChaDiff'] = abs(mindt - maxdt)

            ## DisChage Max,Min 算出
            for i in range(index):
                lpdata = (int((data[i]['DischargeED'] - data[i]['DischargeST']) / 5 * 4)) - (
                    int((data[i]['DischargeED'] - data[i]['DischargeST']) / 5))
                stdt = int((data[i]['DischargeED'] - data[i]['DischargeST']) / 5)
                mindt = 10000.0
                for j in range(lpdata):
                    wdata = float(V_List[(data[i]['DischargeST'] + j + stdt) - 2]) * 1000.0
                    if wdata < mindt:
                        mindt = wdata
                maxdt = -10000
                for j in range(lpdata):
                    wdata = float(V_List[(data[i]['DischargeST'] + j + stdt) - 2]) * 1000.0
                    if wdata > maxdt:
                        maxdt = wdata

                maxmin_data[i]['DisMin'] = mindt
                maxmin_data[i]['DisMax'] = maxdt
                maxmin_data[i]['DisDiff'] = abs(mindt - maxdt)

            objData = self.resultInput(index,finalVoltage,onecyclesteptime,cou_data, op_data, op_diff_data, maxmin_data)
        else:
            ## サイクルエラー
            print("------ Error! not a normal cycle")
            if onecyclesteptime == "":
                onecyclesteptime = 'S'
                objData.append(onecyclesteptime)
            return res,objData

        ### 全表示
        # print("------ All data disp ----- data num=" + str(len(hitcell)))
        if hitcell.count == 0:
            ## 切り出しデータエラー
            print("------ Error! data error")
            if onecyclesteptime == "":
                onecyclesteptime = 'S'
                objData.append(onecyclesteptime)
            return res,objData

        ### 構造体表示
        # print("------ Structure disp -----data num=" + str(index))
        if index != 0:
            res = True  ## エラーなし
        else:
            ## 切り出しデータエラー
            res = False

        return res,objData


    def resultInput(self,num, finalVoltage, onecyclesteptime, cou_data, op_data, op_diff_data, maxmin_data):
        """
        Args:
            num(int): the number of cycle of SD8 measurements
            finalVoltage(str): the final voltage
            onecyclesteptime(str): the step time of final discharge mode at 1st cycle
            cou_data(list): the Coulombic efficiency for each cycle
            op_data(list): the over potential for each cycle
            op_diff_data(list): the difference of over potential for each cycle
            maxmin_data(list): the maximum/minimum of voltage for each cycle

        Returns:
            objData(list): the list of objectives shown in Note

        Note:

            FinalVoltage: Final Voltage

            StepCycle: 1cycle steptime

            CE1: Cycle1 Coulombic efficiency

            CE2: Cycle2 Coulombic efficiency

            CE3: Cycle3 Coulombic efficiency

            CEVE: Cycle2 Cycle3 Coulombic efficiency average

            Cha12mv2, Cha13mv2, Cha23mv2: Cycle2 Charge Voltage

            Dis12mv2, Dis13mv2, Dis23mv2: Cycle2 Discharge Voltage

            OP1_2mv2, OP1_3mv2, OP2_3mv2: Cycle2 over potentiol

            Cha12mv3, Cha13mv3, Cha23mv3: Cycle3 Charge Voltage

            Dis12mv3, Dis13mv3, Dis23mv3: Cycle3 Discharge Voltage

            OP1_2mv3, OP1_3mv3, OP2_3mv3: Cycle3 over potentiol

            DiFF12, DiFF13, DiFF23: Cycle voltage differrence

            minMaxchdiff2, minMaxdisdiff2, minMaxchdiff3, minMaxdisdiff3: min and max for charge and discharge in Cycle2 and Cycle3


        """

        objData = []
        #del objData[:]
        objData.clear()

        objData.append(finalVoltage)        # 最終電圧値セット
        objData.append(onecyclesteptime)    ## １サイクル終了ステップ時間セット

        ## coulomb
        ceve = 0
        for i in range(num):
            objData.append(str("{:.6f}".format(cou_data[i]['CE'])))
            if (i != 0):
                ceve = ceve + cou_data[i]['CE']

        # CE2 とCE3 の平均　を　CEVE　とする
        ceve = ceve / (num - 1)

        objData.append(str("{:.6f}".format(ceve)))

        ## over potential
        for i in range(num-1):
            objData.append(str("{:.6f}".format(op_data[i+1]['Cha1_2mv'])))
            objData.append(str("{:.6f}".format(op_data[i+1]['Cha1_3mv'])))
            objData.append(str("{:.6f}".format(op_data[i+1]['Cha2_3mv'])))
            objData.append(str("{:.6f}".format(op_data[i+1]['Dis1_2mv'])))
            objData.append(str("{:.6f}".format(op_data[i+1]['Dis1_3mv'])))
            objData.append(str("{:.6f}".format(op_data[i+1]['Dis2_3mv'])))
            objData.append(str("{:.6f}".format(op_data[i+1]['OP1_2mv'])))
            objData.append(str("{:.6f}".format(op_data[i+1]['OP1_3mv'])))
            objData.append(str("{:.6f}".format(op_data[i+1]['OP2_3mv'])))

        objData.append(str("{:.6f}".format(op_diff_data[1]['OP_Diff1_2'])))
        objData.append(str("{:.6f}".format(op_diff_data[1]['OP_Diff1_3'])))
        objData.append(str("{:.6f}".format(op_diff_data[1]['OP_Diff2_3'])))

        objData.append(str("{:.6f}".format(maxmin_data[1]['ChaDiff'])))
        objData.append(str("{:.6f}".format(maxmin_data[1]['DisDiff'])))
        objData.append(str("{:.6f}".format(maxmin_data[2]['ChaDiff'])))
        objData.append(str("{:.6f}".format(maxmin_data[2]['DisDiff'])))

        return objData


    def extract_objectives(self,num_objectives, output_folder, p_List):
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

        try:

            # object file load
            filepath = output_folder + "/results.csv"
            with open(filepath) as inf:
                reader = csv.reader(inf)
                objectives_List = [row for row in reader]


            # making log file
            ## スクリプト実行ディレクトリ取得
            current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            resfolder = current_dir + "\\Res"
            if not os.path.exists(resfolder):  # フォルダがなければ生成する
                os.makedirs(resfolder)
            dt_now = datetime.datetime.now()
            resFile = dt_now.strftime('%Y%m%d%H%M%S') + '_res.csv'
            with open(current_dir + "\\Res\\" + resFile, mode="w",newline="") as outf:
                outf.write(self.measFolder+'\n')  ## 測定フォルダ名を書き込む
                ## object Data
                for i in range(len(p_List) - 1):
                    wdata = p_List[i+1] + objectives_List[i]
                    writer = csv.writer(outf)
                    writer.writerow(wdata)

            # add object
            o_List = []

            for i in range(len(p_List)-1):
                insdate = []
                for j in range(num_objectives):
                    objkey = 'objsel' + str(j+1)
                    objmulkey = 'mult' + str(j+1)

                    objidx = self.object_info.get(objkey)
                    objmul = self.object_info.get(objmulkey)
                    objstr = objectives_List[i][objidx]

                    if objstr != 'S':
                        if type(objmul) is int:
                            objdata = math.floor(float(objectives_List[i][objidx])) * objmul
                            objstr = (str)(objdata)
                        else:
                            objData = float(objectives_List[i][objidx]) * objmul
                            objstr = str("{:.6f}".format(objData))
                    insdate.append(objstr)

                if not ('S' in insdate):
                    o_List.append([p_List[i+1][0],insdate])

            res = True

        except:
            res = False

        return res, o_List


    def update_candidate_file(slf,num_objectives, output_file, o_List):
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
                combi_list = c_List[int(o_List[i][0])+1][0:-num_objectives] + o_List[i][1]
                c_List[int(o_List[i][0])+1] = combi_list

            with open(output_file, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(c_List)

            res = True

        except:
            res = False

        return res

    def udp_Receive(self):
        """
        Receiving messages using UDP

        This function DEPEND on robot.

        Returns:
            res (bool): True for success, False otherwise.
        """
        # 受信側アドレスの設定
        # 受信側IP
        SrcIP = "127.0.0.1"
        # 受信側ポート番号
        SrcPort = 11003
        # 受信側アドレスをtupleに格納
        SrcAddr = (SrcIP, SrcPort)
        # バッファサイズ指定
        BUFSIZE = 1024
        res = True

        try:
            # ソケット作成
            udpServSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 受信側アドレスでソケットを設定
            udpServSock.bind(SrcAddr)

            # While文を使用して常に受信待ちのループを実行
            while True:
                # ソケットにデータを受信した場合の処理
                # 受信データを変数に設定
                print("UDP Receive Start")
                data, addr = udpServSock.recvfrom(BUFSIZE)
                # 受信データと送信アドレスを出力
                print(data.decode(), addr)
                break

        except:
            print("UDP Receive Error!")
            res = False

        udpServSock.close()
        return (res)
