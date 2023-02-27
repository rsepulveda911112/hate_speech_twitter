import pandas as pd
import os


def json_filter(file):
    df = pd.read_json(os.getcwd() + file, lines=True)
    df_1 = df[df['label'] == 'VIOLENTO']
    df_1.to_excel(os.getcwd() + '/data/Dataset_VIL_violentos.xlsx', engine='xlsxwriter')

json_filter('/data/Dataset_VIL.json')

