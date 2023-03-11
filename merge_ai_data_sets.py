import pandas as pd
import os
import re
from openpyxl import Workbook
Boris_File_path=r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\AI generate\Boris\Dataset'
Marcell_File_path=r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\AI generate\Marcell\responses'
Thib_File_path=r'C:\Users\tducr\Music\UCL\Term2\COMP0087\project\Data set\AI generate\Thibaud'


def get_data_from_text(File_path):
    data={'Answer':[], 'Question':[], 'Answer_ID':[], 'Model':[]}
    title=False
    os.chdir(File_path)
    #loop over the individual text file and get the
    for file in os.listdir(File_path):
        #check of it is a txtx file
        if file[-4:] == '.txt':
            text = open(file)
            with open(file,'r') as f:
                data['Answer'].append(f.readlines())
            if file[0] == 'q':
                title=True
                data['Question'].append(file[1])
            if title == False:
                data['Answer_ID'].append(file[:-4])
            else:
                data['Answer_ID'].append(re.findall(r'_(\d+).',file))
def get_data_from_excel(File_path):
    data={'Answer':[], 'Question':[], 'Answer_ID':[], 'Model':[]}
    DF=pd.DataFrame()
    for file in os.listdir(File_path):
        if file[-5:] == '.xlsx':
            df=pd.read_excel(File_path + r'\\' + file)
            pd.concat([DF,df])









Marcell_data =get_data_from_text(Boris_File_path)
#Thibaud_data = get_data_from_excel(Thib_File_path)