import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def clean_and_process_csv(df, columns_to_fill_zero=None, apply_strata=False):
    if apply_strata:
        dataset_name_column = df.pop('dataset_name')

    for column in df.columns[1:]:

        df[column] = pd.to_numeric(df[column], errors='coerce')

    non_numeric_columns = df.columns[df.isna().all()]
    df.drop(columns=non_numeric_columns, inplace=True)

    if columns_to_fill_zero:
        for column in columns_to_fill_zero:
            if column in df.columns:

                df[column] = df[column].fillna(0)

    for column in df.columns[1:]:
        if df[column].dtype in ['float64', 'int64']:

            df[column] = df[column].fillna(df[column].mean())

    normalize = True

    if normalize:
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_columns = [col for col in numeric_columns if col != 'survival_time' and col != 'img_name']

        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    if apply_strata and dataset_name_column is not None:
        df['dataset_name'] = dataset_name_column

    return df

def remove_multicollinearity(df, threshold=0.7):
    columns_to_check = [col for col in df.columns if col not in ['img_name', 'survival_time', 'Mace', 'dataset_name']]

    to_drop = []
    duplicate_columns = df[columns_to_check].T.duplicated(keep='first')
    to_drop.extend(df[columns_to_check].columns[duplicate_columns])

    for col in columns_to_check:
        if df[col].nunique() == 1:
            to_drop.append(col)

    correlation_matrix = df[columns_to_check].corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                colname = correlation_matrix.columns[i]
                if colname not in to_drop:
                    to_drop.append(colname)

    df_cleaned = df.drop(columns=to_drop, errors='ignore')
    return df_cleaned

