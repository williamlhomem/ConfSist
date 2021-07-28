import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def create_df(data, sensors=False):
    '''
    Função responsável por criar um dataframe com os sensores informado, usado para as funções gráficas definidas por esse módulo.
    I/O:
        data: um pandas dataframe contendo os dados do pump sensor data;
        sensors: uma lista ou um numpy array contendo o nome das colunas referentes aos sensores de interesse para a plotagem. Caso sensors=False, todos os sensores são plotados;
        return um pandas dataframe contendo os dados que serão utilizados para a plotagem.
    '''
    if sensors:
        df = data[sensors].copy()
        df[['timestamp', 'machine_status']] = data[['timestamp', 'machine_status']].values
    else:
        df = data.copy()
    
    return df

def sensor_plot(data, sensors=False):
    '''
    Função responsável por realizar um lineplot dos sensores contidos no conjunto de dados.
    I/O:
        data: um pandas dataframe contendo os dados do pump sensor data;
        sensors: uma lista ou um numpy array contendo o nome das colunas referentes aos sensores de interesse para a plotagem. Caso sensors=False, todos os sensores são plotados.
    '''
    # cria um df para manipulação e de variáveis auxiliares
    df = create_df(data, sensors=sensors)
    sensor_cols = df.drop(columns=['timestamp', 'machine_status']).columns
    n_sensors = sensor_cols.shape[0]

    # crias dfs filtrados para os rótulos broken e recovering
    b_data = df.groupby('machine_status').get_group('BROKEN')
    r_data = df.groupby('machine_status').get_group('RECOVERING')

    # configuração e plotagem
    fig, ax = plt.subplots(nrows=n_sensors, figsize=(20, 5*n_sensors))
    ax = np.array([ax])
    ax = ax.reshape((n_sensors,))
    iax = 0
    for s in sensor_cols:
        ax[iax].plot_date(x=df['timestamp'], y=df[s], fmt='g', label='Normal')
        ax[iax].plot_date(x=b_data['timestamp'], y=b_data[s], fmt='or', ms=6, label='Broken')
        ax[iax].plot_date(x=r_data['timestamp'], y=r_data[s], fmt='.', c='tab:orange', ms=.5, label='Recovering')
        ax[iax].set_xlabel('Timestamp', size=20)
        ax[iax].set_ylabel(f'{s}', size=20)
        ax[iax].tick_params(labelsize=20)
        ax[iax].grid()
        ax[iax].legend(prop={'size':20})

        iax+=1

def dist_plot(data, sensors=False):
    '''
    Função responsável por plotar as distribuições (KDE plot) dos sensores contidos no conjunto de dados.
    I/O:
        data: um pandas dataframe contendo os dados do pump sensor data;
        sensors: uma lista ou um numpy array contendo o nome das colunas referentes aos sensores de interesse para a plotagem. Caso sensors=False, todos os sensores são plotados.
    '''
    # criação do df e de variáveis auxiliares
    df = create_df(data, sensors=sensors)
    df['b_machine_status'] = data['b_machine_status'].copy()
    sensor_cols = df.drop(columns=['timestamp', 'machine_status', 'b_machine_status']).columns
    n_sensors = sensor_cols.shape[0]

    # criação de dfs filtrados para os rótulos normal e anomaly
    n_data = df.groupby('b_machine_status').get_group('NORMAL')
    a_data = df.groupby('b_machine_status').get_group('ANOMALY')

    # configuração e plotagem
    n_cols = 2
    n_rows = np.ceil(n_sensors/n_cols).astype('int')
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(7*n_cols, 7*n_rows), tight_layout=True)
    ax = ax.reshape((n_cols*n_rows,))
    iax = 0
    for s in sensor_cols:
        p1 = sns.kdeplot(data=n_data, x=s, color='g', ax=ax[iax])
        ax[iax].set_xlabel(f'{s}', size=13)
        ax[iax].set_ylabel(f'Função de densidade: NORMAL', size=13)
        ax[iax].tick_params(labelsize=13)

        axx = ax[iax].twinx()
        p2 = sns.kdeplot(data=a_data, x=s, color='r', ax=axx)
        axx.set_ylabel(f'Função de densidade: ANOMALY', size=13)
        axx.tick_params(labelsize=13)
        
        try:
            ax[iax].legend((ax[iax].get_lines()[0], axx.get_lines()[0]), ('Normal', 'Anomaly'))
        except:
            pass
        ax[iax].grid()

        iax+=1

def boxplot(data, sensors=False):
    '''
    Função responsável por plotar os boxplots dos sensores contidos no conjunto de dados.
    I/O:
        data: um pandas dataframe contendo os dados do pump sensor data;
        sensors: uma lista ou um numpy array contendo o nome das colunas referentes aos sensores de interesse para a plotagem. Caso sensors=False, todos os sensores são plotados.
    '''
    # criação do df e de variáveis auxiliares
    df = create_df(data, sensors=sensors)
    df['b_machine_status'] = data['b_machine_status'].copy()
    sensor_cols = df.drop(columns=['timestamp', 'machine_status', 'b_machine_status']).columns
    n_sensors = sensor_cols.shape[0]

    # configuração e plotagem
    n_cols = 2
    n_rows = np.ceil(n_sensors/n_cols).astype('int')
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(7*n_cols, 7*n_rows), tight_layout=True)
    ax = ax.reshape((n_cols*n_rows,))
    iax = 0
    for s in sensor_cols:
        sns.boxplot(data=df, y=s, x='b_machine_status', ax=ax[iax])
        ax[iax].set_ylabel(f'{s}', size=13)
        ax[iax].set_xlabel('Condição do sistema', size=13)
        ax[iax].tick_params(labelsize=13)
        ax[iax].grid()

        iax+=1

def corr_plot(data, sensors=False, th=False):
    '''
    Função responsável por plotar o gráfico de correlação bivariada entre as variáveis do conjunto de dados e a variável alvo.
    I/O:
        data: um pandas dataframe contendo os dados do pump sensor data;
        sensors: uma lista ou um numpy array contendo o nome das colunas referentes aos sensores de interesse para a plotagem. Caso sensors=False, todos os sensores são plotados;
        th: um float variando de 0 a 1 indicando quando qual o threshold a ser utilizado na plotagem. Caso th=False, a linha não é plotada
    '''
    # criação do df e de variáveis auxiliares
    df = create_df(data, sensors=sensors)
    df['b_machine_status'] = data['b_machine_status'].copy()
    df['b_machine_status'] = df['b_machine_status'].apply(lambda x: 0 if x == 'NORMAL' else 1)
    df.drop(columns=['timestamp', 'machine_status'], inplace=True)
    c_vec = []
    methods = ['pearson', 'spearman', 'pps']

    # criação do df de correlação
    for m in methods:
        if m != 'pps':
            corr = df.corr(method=m).drop(columns='b_machine_status').loc['b_machine_status', :].values
            c_vec.append(corr)
        else:
            pps_df = df.copy()
            df['b_machine_status'] = df['b_machine_status'].astype('object')
            corr_data = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
            corr = corr_data.drop(columns='b_machine_status').loc['b_machine_status', :].values
            c_vec.append(corr)
    
    cols = df.drop(columns='b_machine_status').columns
    
    plot_df = pd.DataFrame()
    for i in range(len(c_vec)):
        aux_df = pd.DataFrame()
        aux_df['cols'] = cols
        aux_df['corr'] = c_vec[i]
        aux_df['method'] = methods[i]

        if plot_df.shape[0] != 0:
            plot_df = pd.concat([plot_df, aux_df])
        else:
            plot_df = aux_df
    
    # configuração e plotagem
    fig, ax = plt.subplots(figsize=(20,10))
    sns.barplot(data=plot_df, x='cols', y='corr', hue='method', ax=ax)
    if th:
        x = [ax.get_xticks()[0]-1, ax.get_xticks()[-1]+1]
        ax.plot(x, [th, th], 'r', label=r'$\rho = $|{}|'.format(th))
        ax.plot(x, [-th, -th], 'r')
        ax.set_xlim(x)
    ax.grid()
    ax.legend(prop={'size':20})
    ax.set_xlabel('Variáveis', size=20)
    ax.set_ylabel('Correalações', size=20)
    ax.tick_params(labelsize=20, rotation=90)

def mvar_corr(data, sensors=False, th=False, report=False):
    '''
    Função responsável por plotar o gráfico de correlação multivariada pela técnica Lasso entre as variáveis do conjunto de dados e a variável alvo.
    I/O:
        data: um pandas dataframe contendo os dados do pump sensor data;
        sensors: uma lista ou um numpy array contendo o nome das colunas referentes aos sensores de interesse para a plotagem. Caso sensors=False, todos os sensores são plotados;
        th: um float variando de 0 a 1 indicando quando qual o threshold a ser utilizado na plotagem. Caso th=False, a linha não é plotada;
        report: um booleano indicando quando se deve plotar o rlatório de classificação do modelo Lasso criado. Caso report=True, a função não plota o gráfico de correlações.
    '''
    # criação do df e de variáveis auxiliares
    df = create_df(data, sensors=sensors)
    df['b_machine_status'] = data['b_machine_status'].copy()
    df['b_machine_status'] = df['b_machine_status'].apply(lambda x: 0 if x == 'NORMAL' else 1)
    df.drop(columns=['timestamp', 'machine_status', 'sensor_15'], inplace=True)
    df.dropna(inplace=True)

    # igualando as classes
    adf = df.groupby('b_machine_status').get_group(1)
    ndf = df.groupby('b_machine_status').get_group(0).sample(n=adf.shape[0])
    df = pd.concat([adf, ndf])

    # criação dos sub datasets X e y
    cols = df.drop(columns='b_machine_status').columns
    X = df[cols].copy()
    y = df['b_machine_status'].copy()

    # regularização do sub conjunto X
    std_scaler = StandardScaler()
    X[cols] = std_scaler.fit_transform(X[cols])

    # criação e treinamento do modelo
    lrgressor = LogisticRegression(penalty='l1', max_iter=1000, solver='liblinear', random_state=42)
    lrgressor.fit(X, y)

    # verificação se o modelo convergiu
    if report:
        yhat = lrgressor.predict(X)
        print(classification_report(y, yhat))
        return None
    
    # extração das importâncias
    importances = lrgressor.coef_[0]
    plot_df = pd.DataFrame(data=np.array([cols, importances]).T, columns=['cols', 'importances'])

    # configuração e plotagem
    fig, ax = plt.subplots(figsize=(20,10))
    sns.barplot(data=plot_df, x='cols', y='importances', ax=ax)
    if th:
        x = [ax.get_xticks()[0]-1, ax.get_xticks()[-1]+1]
        ax.plot(x, [th, th], 'r', label=r'$\rho = $|{}|'.format(th))
        ax.plot(x, [-th, -th], 'r')
        ax.set_xlim(x)
    ax.grid()
    ax.set_xlabel('Variáveis', size=20)
    ax.set_ylabel('Importância', size=20)
    ax.tick_params(labelsize=20, rotation=90)

def redundant_features(data, sensors=False):
    '''
    Função responsável por plotar um mapa de calor apresentando as variáveis redundantes.
    I/O:
        data: um pandas dataframe contendo os dados do pump sensor data;
        sensors: uma lista ou um numpy array contendo o nome das colunas referentes aos sensores de interesse para a plotagem.
    '''
    # criação dodf e de variáveis auxiliares
    df = create_df(data, sensors=sensors)
    df.drop(columns=['timestamp', 'machine_status'], inplace=True)
    corr_df = df.corr()

    # configuração e plotagem
    mask = np.triu(np.ones_like(corr_df, dtype='bool'))
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(data=corr_df, vmin=-1, vmax=1, annot=True, cmap='coolwarm', mask=mask, ax=ax)
    ax.tick_params(labelsize=20)