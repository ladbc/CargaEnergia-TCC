import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics import tsaplots


def multiple_plots(df, regions):
    sns.color_palette("mako", as_cmap=True)
    sns.set(rc={'figure.figsize': (15, 10)})

    df = pd.DataFrame(df)
    dataYearly = df.groupby(df.loc[:, 'Ano'])
    for i in range(0, 10):
        year = 2012 + i
        group = dataYearly.get_group(year)
        for region in regions:
            plt.plot(group.loc[group['ID'] == region]['Data'],
                     group.loc[group['ID'] == region]['Carga_WMed'],
                     label="Subsistema {}".format(region))
            plt.xlabel('Ano', fontsize=18)
            plt.ylabel('Carga WMed', fontsize=18)
        plt.legend(loc='center right')
        plt.title("Carga WMed no ano de {}".format(year), fontsize=20)
        plt.show()


def correlation_plot(data, regionName):
    corr = data.corr()
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                square=True)
    plt.title('Correlação entre variáveis para região {}'.format(regionName), fontsize=20)
    plt.show()


def autocorrelation_plot(data, regionName, lags):
    fig = tsaplots.plot_acf(data['Carga_WMed'], lags=lags)
    plt.ylabel('Autocorrelação', fontsize=18)
    plt.xlabel('Lag', fontsize=18)
    plt.title('Autocorrelação da variável de carga para região {}'.format(regionName), fontsize=20)
    plt.grid()
    plt.show()


def plot_result(predictTest, x_validate, y_validate):
    plt.figure(figsize=(16, 9))
    plt.plot(x_validate[:, 0], predictTest, label="Predicao")
    plt.plot(x_validate[:, 0], y_validate, label="Reais")
    plt.title("Veradeira vs Prevista")
    plt.ylabel("Carga")
    plt.legend(('Verdadeira', 'Prevista'))
    plt.legend()
    plt.show()
