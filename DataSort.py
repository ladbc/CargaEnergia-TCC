import pandas as pd

def sort_data(df, regions):
    df = rename_cols(df)
    df = convert_format(df)
    df = add_info(df)
    regionGroup = get_groups(df, regions)

    return regionGroup, df


def divide_data(df, regions, trainYears, validateYears, testYears):
    trainData = df.loc[df['Ano'].isin(trainYears)]
    validateData = df.loc[df['Ano'].isin(validateYears)]
    testData = df.loc[df['Ano'].isin(testYears)]

    return get_groups(trainData, regions), get_groups(validateData, regions), get_groups(testData, regions)


def rename_cols(df):
    return df.rename(columns={"id_subsistema": "ID",
                              "nom_subsistema": "Subsistema",
                              "din_instante": "Data",
                              "val_cargaenergiamwmed": "Carga_WMed"})


def convert_format(df):
    df.loc[:, 'Data'] = pd.to_datetime(df.loc[:, 'Data'])
    df['ID'] = df['ID'].apply(str)
    df['Subsistema'] = df['Subsistema'].apply(str)

    return df


def add_info(df):
    df['Dia_Semana'] = df['Data'].dt.dayofweek
    df['Dia'] = df['Data'].dt.day
    df['Mes'] = df['Data'].dt.month
    df['Ano'] = df['Data'].dt.year

    return df


def get_groups(df, regions):
    regionIds = df.groupby(df.loc[:, 'ID'])
    regionGroups = []
    for region in regions:
        regionGroups.append(regionIds.get_group(region))
    return regionGroups


def get_years(df):
    return df.groupby(df.loc[:, 'Year'])
