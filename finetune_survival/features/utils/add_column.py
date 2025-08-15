import pandas as pd
import json

def add_new_column_by_search(df, time_dict, column_to_check, column_for_new):
    df[column_for_new] = None

    for index, row in df.iterrows():
        value_to_check = row[column_to_check]
        if value_to_check in time_dict.keys():
            df.at[index, column_for_new] = time_dict[value_to_check]
        else:
            df.at[index, column_for_new] = 'wo'
    df.insert(1, column_for_new, df.pop(column_for_new))
    return df