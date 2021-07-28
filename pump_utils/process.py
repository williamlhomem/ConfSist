import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from scipy.signal import decimate
from tqdm import tqdm

class SL_pipeline(LinearRegression):
    '''
    Objeto responsável pela execução do pipeline necessário para realizar a regressão linear requisitada no notebook statslearning.ipynb, bem como ilustrar seus resultados. Métodos e atributos herdados do sklearn.linear_model.LinearRegression.
    '''

    def results(self, x, y):
        '''
        Método responsável por apresentar a curva de regressão e o gráfico de resíduos da regressão feita.
        I/O:
            x: um numpy array, pandas series ou pandas dataframe contendo os dados de entrada da predição;
            y: um numpy array ou um pandas series contendo os rótulos da predição.
        '''
        # reta real, sem ruído
        yt = 3 + 2*x

        # predição de x
        y_ = self.predict(x.reshape(-1,1))
        rsquared = np.round(r2_score(yt, y_), decimals=4)

        # coeficientes da reta
        w = np.round(self.coef_[0], decimals=3)
        b = np.round(self.intercept_, decimals=3)

        # avalação do resíduo
        res = y - y_

        # configuração e plotagem
        fig, ax = plt.subplots(ncols=2, figsize=(20,10))

        # curva de regressão
        ax[0].plot(x, yt, 'r', label=r'Valor real: $ \beta _0$: 3 e $\beta _1$: 2')
        ax[0].plot(x, y_, 'g', label=r'Valor predito: $ \beta _0$: {} e $\beta _1$: {} ($R^2$: {})'.format(b, w, rsquared))
        ax[0].scatter(x, y)
        ax[0].set_xlabel(f'$x$', size=15)
        ax[0].set_ylabel(f'$y$', size=15)
        ax[0].grid()
        ax[0].legend(prop={'size': 15})

        # gráfico de resíduo
        ax[1].scatter(y_, res)
        ax[1].plot(y_, np.zeros_like(res), 'r')
        ax[1].set_xlabel(f'Predição', size=15)
        ax[1].set_ylabel(f'Resíduos', size=15)
        ax[1].grid()