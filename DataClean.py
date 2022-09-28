import pandas as pd

def clean_data(df):
    regionsData, df = check_for_nan(df)

    return df


# Verifica se o dataset de entrada possui qualquer nan, caso exista
# a linha dever ser imputada com o valor observado na data anterior (LOCF)
def check_for_nan(regions):
    fullset = pd.DataFrame()
    for region in regions:
        nanRows = region[region['Carga_WMed'].isna()]
        if nanRows.shape[0] > 0:
            region['Carga_WMed'] = region['Carga_WMed'].fillna(method='bfill')
        fullset = fullset.append(region)
    print(fullset[['Carga_WMed']])

    return regions, fullset
