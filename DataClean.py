import pandas as pd


def clean_data(df):
    df = check_for_nan(df)
    df = rename_cols(df)
    df = convert_format(df)
    print(df.info())

    return df


def check_for_nan(df):
    # If val_cargaenergiamwmed is nan, delete the row
    return df[df['val_cargaenergiamwmed'].notna()]


def convert_format(df):
    df.loc[:, 'Data'] = pd.to_datetime(df.loc[:, 'Data'])
    df['ID'] = df['ID'].apply(str)
    df['Subsistema'] = df['Subsistema'].apply(str)

    return df


def rename_cols(df):
    return df.rename(columns={"id_subsistema": "ID",
                               "nom_subsistema": "Subsistema",
                               "din_instante": "Data",
                               "val_cargaenergiamwmed": "Carga WMed"})
