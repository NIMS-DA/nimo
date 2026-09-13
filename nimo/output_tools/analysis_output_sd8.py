#Hokuto_SD8

import csv
import datetime
import glob
import os
import re
import shutil
import sys
import time

from pathlib import Path
import pandas as pd
import statistics

EXPECTED_CYCLE_COUNT = 12

class Standard():
    def __init__(self, input_file, output_file, num_objectives, output_folder):
        """Constructor
        
        This function do not depend on robot.

        Args:
            input_file (str): the file for proposals from MI algorithm
            output_file (str): the file for candidates which will be updated in this script
            num_objectives (int): not use
            output_folder (str): the folder where the output files are stored by robot
        """        
        self.input_file = input_file
        self.output_file = output_file
        self.num_objectives = num_objectives
        self.output_folder = output_folder
        self.data_name = None
        self.num_n = None
        self.process_started_date  = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.extract_list = []
        
    class MeasurementCsvData():
        def __init__(self,file_path):
            self.file_path = file_path
            self.is_cycle_target_reached  = True
            self.cha_mAh = []
            self.dis_mAh = []
            self.cha_last_V = []
            self.dis_last_V = []
            self.CE = []
            self.V_diff = []

            self.cha_mAh_default_limit = [0.024,1.440]
            self.dis_mAh_default_limit = [0.024,1.440]
            self.cha_last_V_default_limit= None
            self.dis_last_V_default_limit= None
            self.CE_default_limit= [None,1.000]
            self.V_diff_default_limit= None
                   
        def read_hokuto_csv(self):
            file_path = Path(self.file_path)
            if file_path.is_file():
                self.df = pd.read_csv(
                    self.file_path,
                    encoding='cp932',
                    header=0,
                    skiprows=[1]
                    )
            
            lastcyclenum = (self.df['サイクル'].iloc[-1])

            if lastcyclenum < EXPECTED_CYCLE_COUNT :
                self.is_cycle_target_reached  = False
                
        def add_capacity(self):
            capacity_list = []
            mA = self.df['電流']
            time = self.df['時間']
            mode = self.df['モード']
            
            for i in self.df.index:
                if i != 0 :
                    if mode.iloc[i-1] == 'Rest':
                        capacity_list.append(0)
                    else:
                        capacity_list.append(abs(mA[i-1]) * (float(time[i] - time[i-1]) / 3600) + capacity_list[-1])
                else:
                    capacity_list.append(0)
            self.df['容量'] = capacity_list
            
        def get_value(self):           
            def limit_check(limit,value):
                if not limit == None:
                    if limit[0] == None :
                        limit0 = -float('inf')
                    else:
                        limit0 = limit[0]
                    if limit[1] == None :
                        limit1 = float('inf')
                    else:
                        limit1 = limit[1]
                    if not limit0 <= float(value) <= limit1:
                        self.is_cycle_target_reached  = False
                        
            def value_append(df,array,limit):
                if len(df) >= 1:
                    value = df.iloc[-1].item()
                    array.append(value)                    
                    limit_check(limit,value)
                elif limit == None:
                    array.append(None)
                else:
                    array.append(None)
                    self.is_cycle_target_reached  = False
                return array
            
            def CE_calc(cha_df,dis_df,array,limit):
                if len(cha_df) >= 1 and len(dis_df) >= 1:
                    value = dis_df.iloc[-1].item()/cha_df.iloc[-1].item()
                    array.append(value)
                    limit_check(limit,value)            
                elif limit == None:
                    array.append(None)
                else:
                    array.append(None)
                    self.is_cycle_target_reached  = False
                return array
            
            def V_dif_calc(cha_df,dis_df,array,limit):
                if 2 <= len(dis_df) :                    
                    V_df = pd.concat([cha_df.reset_index(drop=True), dis_df.reset_index(drop=True)], axis=1)
                    dif = V_df.iloc[[int(len(dis_df)/2)]].sum(axis=1).item()
                    array.append(dif)
                elif limit == None:
                    array.append(None)
                else:
                    array.append(None)
                    self.is_cycle_target_reached  = False 
                return array
            
            def limit_None():
                self.cha_mAh = value_append(Charge_mAh_df, self.cha_mAh, None)                    
                self.dis_mAh = value_append(Discharge_mAh_df, self.dis_mAh, None)
                self.CE = CE_calc(Charge_mAh_df, Discharge_mAh_df, self.CE, None)
                self.cha_last_V = value_append(Charge_V_df, self.cha_last_V, None)
                self.dis_last_V = value_append(Discharge_V_df, self.dis_last_V, None)
                self.V_diff = V_dif_calc(Charge_V_df, Discharge_V_df, self.V_diff, None)
                
            def get_each_value():
                self.cha_mAh = value_append(Charge_mAh_df, self.cha_mAh, self.cha_mAh_default_limit)
                self.dis_mAh = value_append(Discharge_mAh_df, self.dis_mAh, self.dis_mAh_default_limit)
                self.CE = CE_calc(Charge_mAh_df, Discharge_mAh_df, self.CE, self.CE_default_limit)
                self.cha_last_V = value_append(Charge_V_df, self.cha_last_V, self.cha_last_V_default_limit)
                self.dis_last_V = value_append(Discharge_V_df, self.dis_last_V, self.dis_last_V_default_limit)
                self.V_diff = V_dif_calc(Charge_V_df, Discharge_V_df, self.V_diff, self.V_diff_default_limit)
            
            Charge_condition = self.df['モード'] == 'Charge'
            Discharge_condition = self.df['モード'] == 'Discharge'
            for i in range(1,EXPECTED_CYCLE_COUNT +1 ):
                cyclecondition = self.df['サイクル'] == i

                Charge_df = self.df[Charge_condition & cyclecondition]
                Discharge_df = self.df[Discharge_condition & cyclecondition]

                Charge_mAh_df = Charge_df['容量']
                Discharge_mAh_df = Discharge_df['容量']

                Charge_V_df = Charge_df['電圧']
                Discharge_V_df= Discharge_df['電圧']
                
                if i == 1 :
                    limit_None()
                else :
                    get_each_value()    
        
    def recieve_exit_message(self):
        try:
            filepath = os.path.join(self.output_folder,'outputend.txt')
            while not(os.path.isfile(filepath)):
                time.sleep(60)            
            os.remove(filepath)
            
            info_path = os.path.join(self.output_folder,'info.txt')
            with open(info_path) as inf:
                self.data_name = inf.read().replace('\r', '').replace('\n', '')
                folder_path = os.path.join(self.output_folder,self.data_name)
            print('The file "outputend.txt" was found.')
            
            return folder_path
        
        except:
            return None
    
    def read_csv(self,path):
        file_path = Path(path)
        if file_path.is_file():
            csv_df = pd.read_csv(
                path,
                encoding='cp932',
                header=0
                )
            return csv_df
        else:
            return None
        
    def initialize_csv_files_df(self,folder_path):
        def channel_number(path: Path | str) -> int:
            name = Path(path).name          
            m = re.search(r'Ch0*([0-9]+)-', name)
            if not m:
                raise ValueError()
            return int(m.group(1))
        
        csv_files = sorted(glob.glob(os.path.join(folder_path, 'Ch*-*.csv')))
        csv_files_df = pd.DataFrame(csv_files, columns=['csv_file'])
        csv_files_df['num'] = [channel_number(csv_file) for csv_file in csv_files]
        csv_files_df = csv_files_df.set_index('num')
        return csv_files_df 
    
    def file_zip(self,folder_path):
        import zipfile
        
        zip_path = os.path.join(os.path.join(os.path.dirname(self.input_file),'data'),self.data_name + '.zip')
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(folder_path):
                for f in files:
                    full_path = Path(root) / f                     
                    arc_name = full_path.relative_to(folder_path) 
                    zf.write(full_path, arc_name)
        print('Data compression completed!')
    
    def save_res_file(self):
        def contact_df(df,lst,col_names):
            df_lst = pd.DataFrame(lst, columns=col_names)
            df = pd.concat([df, df_lst], axis=1)
            return df
        def add_data_to_res_df(res_df,data,data_col):
            res_df = contact_df(res_df,data['cha_mAh'],data_col['cha_mAh'])
            res_df = contact_df(res_df,data['dis_mAh'],data_col['dis_mAh'])       
            res_df = contact_df(res_df,data['cha_last_V'],data_col['cha_last_V'])
            res_df = contact_df(res_df,data['dis_last_V'],data_col['dis_last_V'])
            res_df = contact_df(res_df,data['CE'],data_col['CE'],)
            res_df = contact_df(res_df,data['V_diff'],data_col['V_diff'])
            res_df = contact_df(res_df,data['is_cycle_target_reached'],data_col['is_cycle_target_reached'])
            return res_df
        
        data_col = {
            'cha_mAh' : [f'{i} cha mAh' for i in range(1, EXPECTED_CYCLE_COUNT + 1)],
            'dis_mAh' : [f'{i} dis mAh' for i in range(1, EXPECTED_CYCLE_COUNT + 1)],
            'cha_last_V' : [f'{i} cha last V' for i in range(1, EXPECTED_CYCLE_COUNT + 1)],
            'dis_last_V' : [f'{i} dis last V' for i in range(1, EXPECTED_CYCLE_COUNT + 1)],
            'CE' : [f'{i} CE' for i in range(1, EXPECTED_CYCLE_COUNT + 1)],
            'V_diff' : [f'{i} V diff' for i in range(1, EXPECTED_CYCLE_COUNT + 1)],
            'is_cycle_target_reached'  : ['PASS/Fail'],
            }

        data = {
            'cha_mAh' : [],
            'dis_mAh' : [],
            'cha_last_V' : [],
            'dis_last_V' : [],
            'CE' : [],
            'V_diff' : [],
            'is_cycle_target_reached'  : [],
            }
        
        max_ch_num = len(self.propo_df)
        res_df = pd.DataFrame([self.data_name for _ in range(max_ch_num)],columns=['data_name'])
        
        ch_list = [f'ch{i}' for i in range(1, max_ch_num + 1)]
        res_df['ch'] = ch_list
        res_df = pd.concat([res_df, self.propo_df], axis=1)        

        for extract_data in self.extract_list:
            data['cha_mAh'].append(extract_data.cha_mAh)
            data['dis_mAh'].append(extract_data.dis_mAh)
            data['cha_last_V'].append(extract_data.cha_last_V)
            data['dis_last_V'].append(extract_data.dis_last_V)
            data['CE'].append(extract_data.CE)
            data['V_diff'].append(extract_data.V_diff)
            data['is_cycle_target_reached'].append(extract_data.is_cycle_target_reached)

        res_df = add_data_to_res_df(res_df,data,data_col)
                 
           
        res_path = os.path.join(os.path.dirname(self.output_file),'Res_all')                
        if not os.path.isdir(res_path):
            os.makedirs(res_path)
            
        csv_path = os.path.join(res_path,self.data_name + '_proc' + self.process_started_date + '.csv')
        res_df.to_csv(csv_path,
                      index=False,
                      na_rep='None'
                      )
        print('Save Res File!')
        return res_df
    def update_to_candidate_file(self,res_df):
        
        candidate_file = self.read_csv(self.output_file)
        candidate_path = self.output_file
        
        if self.num_n >= 2:
            agg_dict = {
                '12 CE': 'mean',   
                'PASS/Fail': 'all'
            }
            res_df = res_df.groupby('actions', as_index=False).agg(agg_dict)

        res_df = res_df[res_df['PASS/Fail']].copy()
        res_df = res_df.set_index('actions')
        
        candidate_file.update(res_df)        
        candidate_file.to_csv(candidate_path,
                      index=False,
                      na_rep='None'
                      )
        
        candidate_path = os.path.join(os.path.dirname(self.output_file),'candidate_his')
                
        if not os.path.isdir(candidate_path):
            os.makedirs(candidate_path)
            
        candidate_path = os.path.join(candidate_path,'candidates_' + self.data_name + '_proc' + self.process_started_date + '.csv')
        candidate_file.to_csv(candidate_path,
                      index=False,
                      na_rep='None'
                      )
        print('candidate file update!')
    def perform(self):
        def extract(file_path):
            results = self.MeasurementCsvData(file_path)
            results.read_hokuto_csv()
            results.add_capacity()
            results.get_value()
            return results
        
        print('Start analysis output!')
        
        self.propo_df = self.read_csv(self.input_file)
        max_ch_num = len(self.propo_df)
        folder_path = self.recieve_exit_message()
        self.file_zip(folder_path)        
        csv_files_df = self.initialize_csv_files_df(folder_path)
        self.num_n = int(max_ch_num / len(self.propo_df.drop_duplicates()))
        
        for i in range(1 , max_ch_num + 1):
            if i in csv_files_df.index:
                results = extract(csv_files_df.loc[i,'csv_file'])
                self.extract_list.append(results)
            else:
                self.extract_list.append(None)
        if len(self.extract_list):
            res_df = self.save_res_file()
        self.update_to_candidate_file(res_df)


