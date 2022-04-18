from DataClean import clean_data
from DataExploration import multiple_plots
from DataImport import import_data
from DataSort import sort_data, divide_data
from LSTM import lstm

fileList = ["BaseDeDados/CARGA_ENERGIA_2012.csv",
            "BaseDeDados/CARGA_ENERGIA_2013.csv",
            "BaseDeDados/CARGA_ENERGIA_2014.csv",
            "BaseDeDados/CARGA_ENERGIA_2015.csv",
            "BaseDeDados/CARGA_ENERGIA_2016.csv",
            "BaseDeDados/CARGA_ENERGIA_2017.csv",
            "BaseDeDados/CARGA_ENERGIA_2018.csv",
            "BaseDeDados/CARGA_ENERGIA_2019.csv",
            "BaseDeDados/CARGA_ENERGIA_2020.csv",
            "BaseDeDados/CARGA_ENERGIA_2021.csv"]
regions = ['N', 'NE', 'S', 'SE']
trainYears = [2012, 2013, 2014, 2015, 2016, 2017, 2018]
validateYears = [2019, 2020]
testYears = [2021]

dataFrame = import_data(fileList)
dataFrame = clean_data(dataFrame)
regionsData, fullData = sort_data(dataFrame, regions)
multiple_plots(fullData, regions)
trainData, validateData, testData = divide_data(fullData, regions, trainYears, validateYears, testYears)

lstm(trainData, validateData)

