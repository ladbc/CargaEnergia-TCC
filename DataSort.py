def sort_data(df, regions):
    df = add_info(df)
    regionGroup = get_groups(df, regions)
    return regionGroup, df


def divide_data(df, regions, trainYears, validateYears, testYears):
    trainData = df.loc[df['Year'].isin(trainYears)]
    validateData = df.loc[df['Year'].isin(validateYears)]
    testData = df.loc[df['Year'].isin(testYears)]

    return get_groups(trainData, regions), get_groups(validateData, regions), get_groups(testData, regions)


def add_info(df):
    df['WeekDay'] = df['Data'].dt.dayofweek
    df['Month'] = df['Data'].dt.month
    df['Year'] = df['Data'].dt.year

    return df


def get_groups(df, regions):
    regionIds = df.groupby(df.loc[:, 'ID'])
    regionGroups = []
    for region in regions:
        regionGroups.append(regionIds.get_group(region))
    return regionGroups


def get_years(df):
    return df.groupby(df.loc[:, 'Year'])
