from operator import le
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import ppscore as pps

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from hydro_utils import process

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
        df[['condition', 'exp']] = data[['condition', 'exp']].values
    else:
        df = data.copy()
    
    return df

def timeplots(data, labels):
    '''
    Função responsável por plotar os sinais dos sensores contidos no Hydraulic Systems Dataset. Os sinais plotados são de um experimento aleatório.
    I/O:
        data: um numpy array contendo os dados do Hydraulic Systems Dataset;
        labels: uma lista, um numpy array ou um pandas series contendo os rótulos do conjunto de dados.
    '''
    # Separação dos dados em modos de falhas
    total_failure = data[np.where(labels == 3)]
    low_ef = data[np.where(labels == 20)]
    high_ef = data[np.where(labels == 100)]

    # Criação dos nomes dos sensores
    names = [f'PS{num}' for num in range(1, 7)]
    names = names + [f'FS{num}' for num in range(1, 3)]
    names = names + [f'TS{num}' for num in range(1, 5)]
    names = names + ['EPS1', 'VS1', 'CE', 'CP', 'SE']

    # Criação do vetor de condições, contendo registros de um experimento aleatório.
    condition = [total_failure[np.random.randint(total_failure.shape[0])], 
                low_ef[np.random.randint(low_ef.shape[0])], 
                high_ef[np.random.randint(high_ef.shape[0])]]

    # configuração e plots
    fig = go.Figure()
    timestamp = np.arange(1, 6001)
    window = []

    fig.add_trace(go.Scatter(x=timestamp, y=condition[0][:, 0], marker_color='red', mode='lines', name='Total Failure'))
    fig.add_trace(go.Scatter(x=timestamp, y=condition[1][:, 0], marker_color='purple', mode='lines', name='Low Eficiency'))
    fig.add_trace(go.Scatter(x=timestamp, y=condition[2][:, 0], marker_color='blue', mode='lines', name='High Eficiency'))

    fig.update_layout(xaxis=dict(title_text='Timestamp'), yaxis=dict(title_text=names[0]))
    
    for sensor in range(0, 17):
        window.append(dict(label=names[sensor], method='update', 
        args=[{'y': [condition[0][:, sensor], condition[1][:, sensor], condition[2][:, sensor]]}, {'yaxis.title.text': names[sensor]}]))

    fig.update_layout(updatemenus=[dict(type='dropdown', buttons=window)])
    fig.show()

def dist_plot(data, sensors=False):
    '''
    Função responsável por plotar as distribuições (KDE plot) dos sensores contidos no conjunto de dados.
    I/O:
        data: um pandas dataframe contendo os dados do Hydraulic Sensor Dataset;
        sensors: uma lista ou um numpy array contendo o nome das colunas referentes aos sensores de interesse para a plotagem. Caso sensors=False, todos os sensores são plotados.
    '''
    # criação do df e de variáveis auxiliares
    df = create_df(data, sensors=sensors)
    df.drop(columns='exp', inplace=True)
    sensor_cols = df.drop(columns='condition').columns
    n_sensors = sensor_cols.shape[0]

    # separação do conjunto de dados
    he = df.groupby('condition').get_group(100)
    le = df.groupby('condition').get_group(20)
    tf = df.groupby('condition').get_group(3)

    # amostragem dos dados para reduzir o tempo de processamento
    he = he.sample(n=process.n_sample(he))
    le = le.sample(n=process.n_sample(le))
    tf = tf.sample(n=process.n_sample(tf))

    # configuração e plotagem
    n_cols = 2
    n_rows = np.ceil(n_sensors/n_cols).astype('int')
    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(7*n_cols, 7*n_rows), tight_layout=True)
    ax = ax.reshape((n_cols*n_rows,))
    iax = 0
    for s in sensor_cols:
        sns.kdeplot(data=he, x=s, color='g', label='High efficiency', ax=ax[iax])
        sns.kdeplot(data=le, x=s, color='b', label='Low efficiency', ax=ax[iax])
        sns.kdeplot(data=tf, x=s, color='r', label='Total Failure', ax=ax[iax])
        ax[iax].set_xlabel(f'{s}', size=13)
        ax[iax].set_ylabel(f'Função de densidade', size=13)
        ax[iax].tick_params(labelsize=13)
        ax[iax].grid()
        ax[iax].legend()

        iax+=1

def corr_plot(data, sensors=False, th=False):
    '''
    Função responsável por plotar o gráfico de correlação bivariada entre as variáveis do conjunto de dados e a variável alvo.
    I/O:
        data: um pandas dataframe contendo os dados do Hydraulic Systems Dataset;
        sensors: uma lista ou um numpy array contendo o nome das colunas referentes aos sensores de interesse para a plotagem. Caso sensors=False, todos os sensores são plotados;
        th: um float variando de 0 a 1 indicando quando qual o threshold a ser utilizado na plotagem. Caso th=False, a linha não é plotada
    '''
    # criação do df e de variáveis auxiliares
    df = create_df(data, sensors=sensors)
    df.drop(columns='exp', inplace=True)
    
    c_vec = []
    methods = ['pearson', 'spearman', 'pps']

    # criação do df de correlação
    for m in methods:
        if m != 'pps':
            corr = df.corr(method=m).drop(columns='condition').loc['condition', :].values
            c_vec.append(corr)
        else:
            df['condition'] = df['condition'].astype('object')
            corr_data = pps.matrix(df)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
            corr = corr_data.drop(columns='condition').loc['condition', :].values
            c_vec.append(corr)
    
    cols = df.drop(columns='condition').columns
    
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

def redundant_features(data, sensors=False):
    '''
    Função responsável por plotar um mapa de calor apresentando as variáveis redundantes.
    I/O:
        data: um pandas dataframe contendo os dados do Hydraulic Systems Datasets;
        sensors: uma lista ou um numpy array contendo o nome das colunas referentes aos sensores de interesse para a plotagem.
    '''
    # criação dodf e de variáveis auxiliares
    df = create_df(data, sensors=sensors)
    df.drop(columns=['exp', 'condition'], inplace=True)
    corr_df = df.corr()

    # configuração e plotagem
    mask = np.triu(np.ones_like(corr_df, dtype='bool'))
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(data=corr_df, vmin=-1, vmax=1, annot=True, cmap='coolwarm', mask=mask, ax=ax)
    ax.tick_params(labelsize=15)

def compare_data(data, exp=False):
    '''
    Função responsável por comparar os dados tratados e os dados originais, em relação à taxa de amostragem.
    I/O:
        data: um numpy array contendo os dados do Hydraulic Systems Dataset;
        exp: um inteiro contendo o número de experimento a ser comparado. Caso exp=False, o experimento é sorteado aleatoriamente.
    '''
    # escolha de um experimento arbitrário
    n_exp = np.random.randint(0, data.shape[0])
    if exp:
        n_exp = exp

    # carregamento de um sensor de 1 hz e de 10 hz para verificação do upsampling
    onehz_path = './datasets/hydro/VS1.txt'
    vib = pd.read_csv(onehz_path, delimiter='\t', header=None).loc[n_exp,:]

    tenhz_path = './datasets/hydro/FS1.txt'
    flow = pd.read_csv(tenhz_path, delimiter='\t', header=None).loc[n_exp,:]

    # Separação dos dados após o upsampling
    # A indetificação do sensor pode ser feita por meio da verificação de hydro_utils.process.load_hydroSystem
    up_vib = data[n_exp, :, 13]
    up_flow = data[n_exp, :, 6]

    # configuração e plotagem
    fig, ax = plt.subplots(nrows=2, figsize=(20,10), tight_layout=True)
    tax = [ax[0].twiny(), ax[1].twiny()]

    # plot da eficiência
    ax[0].plot(up_vib, 'b')
    ax[0].set_xlabel('Timestamp [100 hz]', size=15)
    ax[0].set_ylabel('Fator de eficiência [%]', size=15)
    ax[0].tick_params(labelsize=15)
    ax[0].grid()

    tax[0].plot(vib, 'r')
    tax[0].set_xlabel('Timestamp [1 hz]', size=15)
    tax[0].legend((ax[0].get_lines()[0], tax[0].get_lines()[0]), ('Upsampling', 'Original'))
    tax[0].tick_params(labelsize=15)

    # plot da vazão
    ax[1].plot(up_flow, 'b')
    ax[1].set_xlabel('Timestamp [100 hz]', size=15)
    ax[1].set_ylabel('Vazão [L/min]', size=15)
    ax[1].tick_params(labelsize=15)
    ax[1].grid()

    tax[1].plot(flow, 'r')
    tax[1].set_xlabel('Timestamp [1 hz]', size=15)
    tax[1].legend((ax[0].get_lines()[0], tax[1].get_lines()[0]), ('Upsampling', 'Original'))
    tax[1].tick_params(labelsize=15)