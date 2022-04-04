df_2019.loc[:, 'din_instante'] = pd.to_datetime(df_2019.loc[:,'din_instante'])
region = df_2019.groupby(df_2019.loc[:,'id_subsistema'])

figure, axis = plt.subplots(2, 2)

df = region.get_group('SE').loc[:, ('din_instante', 'val_cargaenergiamwmed')]
df.plot(x='din_instante', y='val_cargaenergiamwmed', kind='line')

plt.show()