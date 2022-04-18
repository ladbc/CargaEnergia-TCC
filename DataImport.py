import pandas as pd


def import_data(files):
    completeData = pd.DataFrame()

    for file in files:
        data = pd.read_csv(file, header=[0], sep=";")
        completeData = completeData.append(data, ignore_index=True)

    return completeData
