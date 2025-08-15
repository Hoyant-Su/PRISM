
import pandas as pd

def remove_specified_columns_and_save(csv_file, columns_to_remove):
    df = pd.read_csv(csv_file)
    df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')
    df_cleaned.to_csv(csv_file, index=False)
    print(f"Columns {columns_to_remove} have been removed and the file has been updated.")

