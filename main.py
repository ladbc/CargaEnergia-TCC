from LSTMGRU import lstmgru
from DataClean import clean_data
from DataExploration import multiple_plots
from DataImport import import_data
from DataSort import sort_data, divide_data
from LSTM import lstm
from LSTMFF import lstmff
import seaborn as sns
import matplotlib.pyplot as plt

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


# Importa dados a partir de arquivos csv por ano
dataFrame = import_data(fileList)
# Ordena colunas, adiciona informacoes e separa por grupos
regionsData, fullData = sort_data(dataFrame, regions)
# Confere dataset para NaNs por subgrupos
clean_data = clean_data(regionsData)
# Plot inicial dos dados divididos por ano
multiple_plots(fullData, regions)
# Separacao dos dados em conjuntos de treino teste e validacao
trainData, validateData, testData = divide_data(clean_data, regions, trainYears, validateYears, testYears)

# Processo de treino e teste LSTM Bidirecional
lstm(trainData, validateData, testData)
# Processo de treino e teste LSTM Feed Forward
lstmff(trainData, validateData, testData)
# Processo de treino e teste GRU
lstmgru(trainData, validateData, testData)
