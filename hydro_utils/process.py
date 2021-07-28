import os

import numpy as np
import pandas as pd

from tqdm import tqdm

def load_hydroSystem(file_path):
    '''
    Função responsável pelo processamento e importação dos dados do Condition Monitoring of Hydraulic Systems Dataset
    I/O:
        path: uma string contendo o diretório onde os conjuntos de dados dos sensores estão contidos;
        return: um Numpy Array de formato ((nº de instâncias, timestamp, features), label)
    '''

    # Listagem dos arquivos contendo os dados dos sensores
    load_names = os.listdir(file_path)
    load_names.remove('description.txt')
    load_names.remove('documentation.txt')

    # Indexição das colunas para o upsamplig das variáveis com maior taxa de amostragem
    cols_1 = np.arange(0, 6000, 100)
    cols_10 = np.arange(0, 6000, 10)

    # Importação dos dados contidos nos arquivos ".txt"
    # Features
    pressure = []
    flow = []
    temp = []

    print('Carregamento dos conjuntos de dados:')
    for name in tqdm(load_names):
        if 'PS' in name and name != 'EPS1.txt':
            ps = pd.read_csv(f'{file_path}{name}', delimiter='\t', header=None)
            pressure.append(ps)
        elif 'FS' in name:
            aux = pd.read_csv(f'{file_path}{name}', delimiter='\t', header=None)
            fs = pd.DataFrame(data=np.nan*np.ones((aux.shape[0], 6000)))
            fs[cols_10] = aux.values
            fs = fs.interpolate(axis='columns')
            flow.append(fs)
        elif 'TS' in name:
            aux = pd.read_csv(f'{file_path}{name}', delimiter='\t', header=None)
            t = pd.DataFrame(data=np.nan*np.ones((aux.shape[0], 6000)))
            t[cols_1] = aux.values
            t = t.interpolate(axis='columns')
            temp.append(t)

    eps = pd.read_csv(f'{file_path}EPS1.txt', delimiter='\t', header=None)
    vs = pd.read_csv(f'{file_path}VS1.txt', delimiter='\t', header=None)
    ce = pd.read_csv(f'{file_path}CE.txt', delimiter='\t', header=None)
    cp = pd.read_csv(f'{file_path}CP.txt', delimiter='\t', header=None)
    se = pd.read_csv(f'{file_path}SE.txt', delimiter='\t', header=None)

    aux_dfs = [vs, ce, cp, se]
    mod_dfs = []
    for df in aux_dfs:
        aux = df.copy()
        aux_df = pd.DataFrame(data=np.nan*np.ones((aux.shape[0], 6000)))
        aux_df[cols_1] = aux.values
        aux_df = aux_df.interpolate(axis='columns')
        mod_dfs.append(aux_df)

    # Labels
    labels = pd.read_csv(f'{file_path}profile.txt', delimiter='\t', header=None)
    labels = labels[0].copy()

    # Concatenação dos dados
    data = []
    print('Processamento dos dados:')
    for cycle in tqdm(range(2205)):
        example = np.c_[
            pressure[0].loc[cycle, :].values,
            pressure[1].loc[cycle, :].values,
            pressure[2].loc[cycle, :].values,
            pressure[3].loc[cycle, :].values,
            pressure[4].loc[cycle, :].values,
            pressure[5].loc[cycle, :].values,
            flow[0].loc[cycle, :].values,
            flow[1].loc[cycle, :].values,
            temp[0].loc[cycle, :].values,
            temp[1].loc[cycle, :].values,
            temp[2].loc[cycle, :].values,
            temp[3].loc[cycle, :].values,
            eps.loc[cycle, :].values,
            mod_dfs[0].loc[cycle, :].values,
            mod_dfs[1].loc[cycle, :].values,
            mod_dfs[2].loc[cycle, :].values,
            mod_dfs[3].loc[cycle, :].values]

        data.append(example)

    return np.array(data), labels

def create_hydrodf(array, labels):
    '''
    Função responsável por organizar o conjunto de dados Hydraulic Systems Dataset em uma forma tabular.
    I/O:
        array: numpy array contendo os dados em um formato tridimensional;
        labels: lista, numpy array ou pandas series contendo os rótulos do conjunto de dados.
    '''
    df = pd.DataFrame()
    label_exp = []
    label = []
    i = 0

    # concatenação dos dados
    for exp in tqdm(array):
        df = pd.concat([df, pd.DataFrame(exp)], axis='index')
        label_exp.append((i+1)*np.ones(exp.shape[0]))
        label.append(labels[i]*np.ones(exp.shape[0]))
        i += 1

    # nomeamento das colunas
    names = [f'PS{num}' for num in range(1, 7)]
    names = names + [f'FS{num}' for num in range(1, 3)]
    names = names + [f'TS{num}' for num in range(1, 5)]
    names = names + ['EPS1', 'VS1', 'CE', 'CP', 'SE']

    df.columns = names

    # identificação dos experimentos
    label_exp = np.array(label_exp)
    label_exp = label_exp.reshape((label_exp.shape[0]*label_exp.shape[1]))
    df['exp'] = label_exp.astype('int')

    # adição dos rótulos
    label = np.array(label)
    label = label.reshape((label.shape[0]*label.shape[1]))
    df['condition'] = label.astype('int')

    return df

def n_sample(df):
    '''
    Função para calcular o tamanho da amostra de um grupo pela abordagem de Yamane com 99% de intervalo de confiança.
    I/O:
        df: um pandas dataframe que será amostrado;
        return um inteiro indicando o tamanho da amostra
    '''
    N = df.shape[0]
    e = 0.01
    return int(N/(1+N*e**2)) + 1 