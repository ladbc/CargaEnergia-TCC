import pandas as pd

df_2019 = pd.read_csv("BaseDeDados/CARGA_ENERGIA_2019.csv", header=[0], sep=",")
df_2020 = pd.read_csv("BaseDeDados/CARGA_ENERGIA_2020.csv")
df_2021 = pd.read_csv("BaseDeDados/CARGA_ENERGIA_2021.csv")
df_2022 = pd.read_csv("BaseDeDados/CARGA_ENERGIA_2022.csv")


#df_2019.iloc[:, 3] = pd.to_datetime(df_2019.iloc[:,3])
print(df_2019.head())

