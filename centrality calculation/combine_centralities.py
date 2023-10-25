from operator import index
import os
import pandas as pd


def extract_week(s:str)->str:
    return s.split('_')[0]


def combine_centralities(s:str):

    file_list = [file for file in os.listdir(s) if file.endswith('.csv')]
    week_list = [extract_week(week) for week in file_list]

    df = pd.DataFrame()

    for week, file in zip(week_list,file_list):
        newdf = pd.read_csv(f"Result/{file}")
        newdf.eigenvector_centrality = abs(newdf.eigenvector_centrality)
        newdf['week'] = week
        df = df.append(newdf, ignore_index=True)

    df.to_csv('Full_centrality.csv')

    return df
