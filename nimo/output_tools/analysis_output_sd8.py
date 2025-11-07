import csv
import time
import sys
import datetime
import os
import shutil

class SD8():
    """Class of Standard

    This class can perform analysis of outputs from robot.

    """

    def __init__(self, input_file, output_file, num_objectives, output_folder,num_n = None):
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
        self.data_name = None
        self.process_started_date  = datetime.datetime.now()
        self.num_n = num_n


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
        
        res, p_List = self.load_data()

        if res == False:
            print("ErrorCode: error in load_data function")
            sys.exit()

        res, o_List = self.extract_objectives(p_List)

        if res == False:
            print("ErrorCode: error in extract_objectives function")
            sys.exit()

        res = self.update_candidate_file(o_List)

        if res == False:
            print("ErrorCode: error in update_candidate_file function")
            sys.exit()


        print("Finish analysis output!")

        return "True"



    def load_data(self):
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
            with open(self.input_file) as inf:
                reader = csv.reader(inf)
                p_List = [row for row in reader]
                
            res = True

        except:
            res = False

        return res, p_List 



    def recieve_exit_message(self):
        """Recieving exit message from machine

        This function DEPENDS on robot.

        Args:
            output_folder (str): the folder where the results by machine are stored

        Returns:
            res (bool): True for success, False otherwise.

        
        """
        try:
            filepath = os.path.join(self.output_folder,'outputend.txt')
            
            while not(os.path.isfile(filepath)):
                
                time.sleep(60)

            os.remove(filepath)

            res = True


        except:
            res = False
        
        res = True
        return res



    def extract_objectives(self, p_List):    
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
        
        class Data():            
            def __init__(self, data = "header"):
                #想定している最大サイクル数を記載しておく
                self.set_max_cycle_num = 17
                
                if data == "header":
                    self.res_hedder_set()
                    return
                
                self.column_num = self.ExtractColumnNum(data)
                self.data = self.mAh_calc(data)
                    
                self.last_cycle_num = self.data[-1][ self.column_num.cycle ]            
                    
                self.extract_set()
                self.extract_data()
                    
                self.limit_set()
                self.prepare_o_list()
                
            def prepare_o_list(self):                
                try:
                    #candidatesに書き込むobjectの指定
                    self.o_list =[self.coulombic_efficiency[10]]
                    
                except:
                    self.o_list = None
                
                
                self.all_o_list = []
                tmp_list = []
                #RESファイルに書き込むobjectの順番の指定
                tmp_list.append(self.cha_last_mAh)
                tmp_list.append(self.dis_last_mAh)
                tmp_list.append(self.cha_last_v)
                tmp_list.append(self.dis_last_v)
                tmp_list.append(self.coulombic_efficiency)
                tmp_list.append(self.half_capa_V_diff)

                #データのない箇所を埋めておく
                for l in tmp_list:
                    self.all_o_list += [l[i] if i <= len(l)-1 else None for i in range (self.set_max_cycle_num) ]                    
                self.all_o_list += [self.Rint]
                self.all_o_list += [self.RR]
                self.all_o_list += [self.pass_fail]
                return               
                
            def extract_set(self):               
                #取得する値を入れるlistを準備
                self.cha_last_mAh = []
                self.dis_last_mAh = []
                self.cha_last_v = []
                self.dis_last_v = []
                self.half_capa_V_diff = []
                self.coulombic_efficiency = []
                self.Rint = None
                self.RR = None
                self.pass_fail = True
                return
            
            
            
            def limit_set(self):
                def limit_check(res_list,limit_list):
                            
                    if len(res_list) < len(limit_list):
                        self.pass_fail = False
                        return
                    for index,limit in enumerate(limit_list):                    
                        if not res_list[index]:
                            continue
                        elif not (limit[0] <= float(res_list[index]) <= limit[1]):
                            self.pass_fail = False
                    return
                
                #いったんすべての上限下限リミットを無限に設定。
                cha_last_mAh_limit = [[-float('inf'),float('inf')] for i in range(self.set_max_cycle_num)]
                dis_last_mAh_limit = [[-float('inf'),float('inf')] for i in range(self.set_max_cycle_num)]
                cha_last_v_limit = [[-float('inf'),float('inf')] for i in range(self.set_max_cycle_num)]                
                dis_last_v_limit = [[-float('inf'),float('inf')] for i in range(self.set_max_cycle_num)]                
                hcVdiff_limit = [[-float('inf'),float('inf')] for i in range(self.set_max_cycle_num)]                
                ce_limit = [[-float('inf'),float('inf')] for i in range(self.set_max_cycle_num)]
                Rint_RR_limit = [[-float('inf'),float('inf')],[-float('inf'),float('inf')]]
                
                #範囲設定(下限,上限)
                """
                下限のみ設定する場合は上限にfloat('inf')
                上限のみ設定する場合は下限に-float('inf')
                を入れておく。                
                """
                cha_last_mAh_limit[1] = [0.024,0.300]
                cha_last_mAh_limit[2] = [0.024,0.300]
                cha_last_mAh_limit[3] = [0.024,0.300]
                cha_last_mAh_limit[4] = [0.024,0.300]
                cha_last_mAh_limit[5] = [0.024,0.300]
                cha_last_mAh_limit[6] = [0.024,0.300]
                cha_last_mAh_limit[7] = [0.024,0.300]
                cha_last_mAh_limit[8] = [0.024,0.300]
                cha_last_mAh_limit[9] = [0.024,0.300]
                cha_last_mAh_limit[10] = [0.024,0.300]
                cha_last_mAh_limit[11] = [0.024,0.300]

                dis_last_mAh_limit[1] = [0.024,0.300]
                dis_last_mAh_limit[2] = [0.024,0.300]
                dis_last_mAh_limit[3] = [0.024,0.300]
                dis_last_mAh_limit[4] = [0.024,0.300]
                dis_last_mAh_limit[5] = [0.024,0.300]
                dis_last_mAh_limit[6] = [0.024,0.300]
                dis_last_mAh_limit[7] = [0.024,0.300]
                dis_last_mAh_limit[8] = [0.024,0.300]
                dis_last_mAh_limit[9] = [0.024,0.300]
                dis_last_mAh_limit[10] = [0.024,0.300]
                dis_last_mAh_limit[11] = [0.024,0.300]

                ce_limit[1] = [-float('inf'),1.000]
                ce_limit[2] = [-float('inf'),1.000]
                ce_limit[3] = [-float('inf'),1.000]
                ce_limit[4] = [-float('inf'),1.000]
                ce_limit[5] = [-float('inf'),1.000]
                ce_limit[6] = [-float('inf'),1.000]
                ce_limit[7] = [-float('inf'),1.000]
                ce_limit[8] = [-float('inf'),1.000]
                ce_limit[9] = [-float('inf'),1.000]
                ce_limit[10] = [-float('inf'),1.000]
                ce_limit[11] = [-float('inf'),1.000]

                Rint_RR_limit[0] = [-float('inf'),0] #内部抵抗
                Rint_RR_limit[1] = [0.950,float('inf')] #決定係数
                
                limit_check(self.cha_last_mAh,cha_last_mAh_limit)
                limit_check(self.dis_last_mAh,dis_last_mAh_limit)
                limit_check(self.cha_last_v,cha_last_v_limit)
                limit_check(self.dis_last_v,dis_last_v_limit)
                limit_check(self.half_capa_V_diff,hcVdiff_limit)
                limit_check(self.coulombic_efficiency,ce_limit)
                
                limit_check([self.Rint,self.RR],Rint_RR_limit)
                
                return
                
            def extract_data(self):
                import numpy as np
                #値抽出部
                #サイクル1から順番に
                for cycle in range(1,int(self.last_cycle_num) + 1):
                    #charge部分,discharge部分を抽出
                    cha = [data for data in self.data if all ([int(data[self.column_num.cycle]) == cycle, data[self.column_num.mode].upper() == 'CHARGE'])]
                    dis = [data for data in self.data if all ([int(data[self.column_num.cycle]) == cycle, data[self.column_num.mode].upper() == 'DISCHARGE'])]

                    #charge,dischargeの最終電圧と最終容量を取得
                    if cha:
                        self.cha_last_mAh.append(cha[-1][self.column_num.mAh])
                        self.cha_last_v.append(cha[-1][self.column_num.volt])
                    else:
                        self.cha_last_mAh.append(None)
                        self.cha_last_v.append(None)
                    if dis:
                        self.dis_last_mAh.append(dis[-1][self.column_num.mAh])
                        self.dis_last_v.append(dis[-1][self.column_num.volt])
                    else:
                        self.dis_last_mAh.append(None)
                        self.dis_last_v.append(None)

                    #charge,dischargeが行われていることを確認しクーロン効率を計算。
                    if all([cha,dis]):
                        if int(dis[-1][self.column_num.step]) > int(cha[-1][self.column_num.step]):
                            self.coulombic_efficiency.append(str(float(dis[-1][self.column_num.mAh]) / float(cha[-1][self.column_num.mAh])))                            
                        elif int(cha[-1][self.column_num.step]) > int(dis[-1][self.column_num.step]):
                            self.coulombic_efficiency.append(str(float(cha[-1][self.column_num.mAh]) / float(dis[-1][self.column_num.mAh])))
                        else:
                            self.coulombic_efficiency.append(None)
                            
                        cha_half = [ data for data in cha if float(data[ self.column_num.mAh ]) >= ( float(dis[-1][self.column_num.mAh]) / 2 ) ]
                        dis_half = [ data for data in dis if float(data[ self.column_num.mAh ]) >= ( float(dis[-1][self.column_num.mAh]) / 2 ) ]
                        if all([cha_half,dis_half]):
                            self.half_capa_V_diff.append( str(float(cha_half[0][ self.column_num.volt ]) - float(dis_half[0][ self.column_num.volt ])))
                        else:
                            self.half_capa_V_diff.append(None)
                    else:
                        self.coulombic_efficiency.append(None)
                        self.half_capa_V_diff.append(None)
                    
                #内部抵抗の計算
                if int(self.last_cycle_num) >= self.set_max_cycle_num :
                    x = np.array([ 0.2 , 0.5 , 1.0 ])
                    y = np.array([float(self.dis_last_v[13]) , float(self.dis_last_v[14]) , float(self.dis_last_v[15])])
                    res=np.polyfit(x, y, 1)
                    Rint= res[0] #内部抵抗
                    y1 = np.poly1d(res)(x) 
                    RR = (np.corrcoef(y,y1)[1,0])**2 #決定係数

                    self.Rint = Rint
                    self.RR = RR
                return
                
            def mAh_calc(self,data):
                #測定データの容量値がAhで記載されていることがあるのでmAhに再計算
                del data[0]
                del data[0]
                new_data = []
                pre_time = 0.0
                pre_mA = 0.0
                pre_capacity = 0.0
                for row in data:
                    
                    if row[int(self.column_num.mode)].upper() == 'REST':                        
                        capacity = 0.0                        
                    else:
                        capacity = ((abs(pre_mA)) * (float(row[self.column_num.time]) - pre_time) / 3600) + pre_capacity                        
                    pre_time = float( row[ self.column_num.time])
                    pre_mA = abs(float(row[self.column_num.ampere]) * self.column_num.ampere_fix)
                    pre_capacity = capacity
                    row.append(str(capacity))
                    new_data.append(row)                
                return new_data
            
            def res_hedder_set(self):
                #RESファイルのヘッダー設定
                self.o_list_hedder = []
                hedder = [[] for l in range(9)]
                for i in range(self.set_max_cycle_num):
                    hedder[0].append(str(i+1) + ' cha mAh')
                    hedder[1].append(str(i+1) + ' dis mAh')
                    hedder[2].append(str(i+1) + ' cha last V')
                    hedder[3].append(str(i+1) + ' dis last V')
                    hedder[4].append(str(i+1) + ' CE')
                    hedder[5].append(str(i+1) + ' V diff')
                #サイクル数に左右されないobjectはここ
                hedder[6].append('R int')
                hedder[7].append('R^2')
                hedder[8].append('PASS/Fail')
                
                for h in  hedder:
                    self.o_list_hedder += h
                return
                    
            class ExtractColumnNum():
                def __init__(self, data):
                    #列を探して列番号を記録
                    header = [data[0],data[1]]
                    header_t = [list(x) for x in zip(*header)]                              
                
                    for index, column in enumerate(header_t):
                        #必要な列のみ取得                        
                        if any([column[0] == '時間',column[0].upper() == 'TIME']):
                            self.time = index
                            
                        if any([column[0] == '電圧',column[0].upper() == 'VOLT',column[0].upper() == 'VOLTAGE',column[0].upper() == 'V']):
                            self.volt = index
                            
                            if column[1].upper() == 'MV':
                                self.volt_fix = 1/1000
                            else:
                                self.volt_fix = 1
                                
                        if any([column[0] == '電流',column[0].upper() == 'AMPERE',column[0].upper() == 'A']):
                            self.ampere = index
                            
                            if column[1].upper() == 'A':
                                self.ampere_fix = 1000
                            else:
                                self.ampere_fix = 1
                                
                        if any([column[0] == 'サイクル',column[0] == 'ｻｲｸﾙ',column[0].upper() == 'CYCLE']):
                            self.cycle = index
                            
                        if any([column[0] == 'モード',column[0] == 'ﾓｰﾄﾞ',column[0].upper() == 'MODE']):
                            self.mode = index
                            
                        if any([column[0] == 'ステップ',column[0] == 'ｽﾃｯﾌﾟ',column[0].upper() == 'STEP']):
                            self.step = index

                        #容量値は再計算したものを使用するので最終列となる
                        self.mAh = -1
                            
                    return
            #classはここまで
                
        def calc_average(o_List):            
            import numpy as np
            Newlist = []
            o_List = sorted(o_List)

            pre_num = None
            for row in o_List:
                if pre_num == row[0]:
                    continue
                if sum([l.count(row[0]) for l in o_List]) >= self.num_n:
                    ave_list = list(np.float64( [l[1:] for l in o_List if l[0] == row[0]] ))
                    n = np.mean(ave_list, axis=0).tolist()[0]
                    
                    n = [str(l) for l in n ]
                    Newlist.append([row[0],n])
                    pre_num = row[0]
            
            return Newlist
        
        def make_res_file(o_all_List):
            all_res_file_name = (self.data_name + '(' + str(self.process_started_date.strftime('%Y%m%d%H%M%S') + ')_all.csv'))
            all_res_dir = os.path.join(os.path.dirname(self.output_file),'Res_all')
            all_res_path = os.path.join(all_res_dir,all_res_file_name)
           
            if not os.path.exists(all_res_dir):
                os.makedirs(all_res_dir)
                
            o_all_List = [ ['None' if d == None else d for d in row] for row in o_all_List]
            o_all_List = [ ['PASS' if d == True else d for d in row] for row in o_all_List]
            o_all_List = [ ['Fail' if d == False else d for d in row] for row in o_all_List]
                
            data = Data()
            o_list_hedder = ['data_name','ch','index'] + p_List[0][1:] + data.o_list_hedder
            del data
            with open(all_res_path, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(o_list_hedder)
                writer.writerows(o_all_List)
            return
        import zipfile
        try:
        
            o_List = []
            o_all_List = []
            o_list_hedder = []
            info_path = os.path.join(self.output_folder,'info.txt')
            with open(info_path) as inf:
                self.data_name = inf.read().replace('\r', "").replace('\n', "")
            folder_path = os.path.join(self.output_folder,self.data_name)
            
            datapath = os.path.join(os.path.dirname(self.output_file),'data')
            
            if not os.path.exists(datapath):
                os.makedirs(datapath)
            zip_file_path = os.path.join(datapath, self.data_name + '.zip')
            with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zip_file.write(file_path, os.path.relpath(file_path, folder_path))
            for index, p_num, in enumerate(p_List):
                status = True
                if index == 0 :
                    continue
                elif p_num[0] == '':
                    continue
                file_name = 'Ch' + str( index ).rjust(3, '0') + '-001.csv'
                file_path = os.path.join(folder_path, file_name)
                print(file_path)
                if not os.path.isfile(file_path):
                    status = False
                else:
                    with open(file_path, encoding='Shift_JIS') as inf:
                        reader = csv.reader(inf)
                        csv_data = [row for row in reader]
                    if len(csv_data) <= 2 :
                        status = False
                    else:                            
                        data = Data(csv_data) 
                        o_all_List.append([self.data_name] + ['ch' + str( index )] + p_num + data.all_o_list)
                    
                        if data.all_o_list[-1] == True:
                            o_List.append([p_num[0],data.o_list])
                        del data
                     
                if status == False:
                    o_all_List.append([self.data_name] + ['ch' + str( index )] + p_num)
           
            if self.num_n>=2:
                o_List = calc_average(o_List)                
            res = True
            make_res_file(o_all_List)
                           
        
            
        except Exception as e:
            raise

            res = False
            
        return res, o_List


    def update_candidate_file(self, o_List):
        """Updating candidates

        This function do not depend on robot.

        Args:
            num_objectives (int): the number of objectives
            output_file (str): the file for candidates
            o_List (list[float]): the list of objectives

        Returns:
            res (bool): True for success, False otherwise.

        """
        
        candidate_his_dir = os.path.join(os.path.dirname(self.output_file),'candidate_his')
        candidate_his_path = os.path.join(candidate_his_dir,('candidate_his_' + self.data_name + '(' + str(self.process_started_date.strftime('%Y%m%d%H%M%S') + ').csv')))
        if not os.path.exists(candidate_his_dir):
            os.makedirs(candidate_his_dir)
        
        try:
            with open(self.output_file) as inf:
                reader = csv.reader(inf)
                c_List = [row for row in reader]
                
            for o_List_row in o_List:
                combi_list = c_List[int(o_List_row[0])+1][0:-self.num_objectives] + o_List_row[1]
                c_List[int(o_List_row[0])+1] = combi_list

            with open(self.output_file, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(c_List)
                
            with open(candidate_his_path, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerows(c_List)

            res = True

        except:
            res = False
            
        return res
    
    

