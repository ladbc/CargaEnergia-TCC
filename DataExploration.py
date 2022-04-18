import matplotlib.pyplot as plt
import pandas as pd


def multiple_plots(df, regions):
    df = pd.DataFrame(df)
    dataYearly = df.groupby(df.loc[:, 'Year'])
    for i in range(0, 10):
        year = 2012 + i
        group = dataYearly.get_group(year)
        for region in regions:
            plt.plot(group.loc[group['ID'] == region]['Data'],
                     group.loc[group['ID'] == region]['Carga WMed'],
                     label="Subsistema {}".format(region))
            plt.xlabel('Ano')
            plt.ylabel('Carga WMed')
        plt.legend()
        plt.title("Carga WMed no ano de {}".format(year))
        plt.show()
