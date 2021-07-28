# importações
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.relational import scatterplot

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

class baseline_model():
    '''
    Classe responsável pelo modelo baseline, que utiliza apenas um threshold da temperatura do fluido hidráulico para realizar a prediçãoda condição do cooler.
    '''
    def __init__(self, high, failure):
        '''
        Método responsável pelo instanciamento da classe.
        I/O:
            high: um float indicando o limite usado para considerar a condição do cooler como alta eficiência;
            failure: um float indicando o limite usado para considerar a condição do cooler como falha crítica
        '''
        self.limits = np.array([high, failure])

    def predict(self, temps):
        '''
        Método responsável por realizar a predição do modelo.
        I/O:
            temps: lista, um numpy array ou um pandas series contendo as temperaturas do fluido hidráulico.
        '''
        preds = []
        for temp in temps:
            if temp < self.limits[0]:
                predict = 100
            elif temp > self.limits[1]:
                predict = 3
            else:
                predict = 20
            
            preds.append(predict)
        
        return np.array(preds)

    def report(self, x, y):
        '''
        Método responsável por avaliar o modelo baseline.
        I/O:
            x: lista, numpy array ou pandas series contendo as temperaturas do fluido hidráulico.
            y: lista, numpy array ou pandas series contendo os rótulos verdadeiros.
        '''
        preds = self.predict(x)
        print(classification_report(y, preds))

class RF_classifier(RandomForestClassifier):
    '''
    Classe responsável pela criação, treinamento e avaliação da performance de um modelo classificador do tipo Random Forest.
    '''
    def train(self, x, y):
        '''
        Método responsável pelo treinamento do modelo.
        I/O:
            x: numpy array do tipo (n_samples, timestamp, features) contendo as entradas do modelo;
            y: lista, numpy array ou pandas series contendo os rótulos do modelo.
        '''
        reshaped_x = x.reshape((x.shape[0], -1))
        self.fit(reshaped_x, y)
    
    def report(self, x, y, cross=False):
        '''
        Método responsável pelo treinamento do modelo.
        I/O:
            x: numpy array do tipo (n_samples, timestamp, features) contendo as entradas do modelo;
            y: lista, numpy array ou pandas series contendo os rótulos do modelo;
            cross: um booleano que indica quando se deve fazer validação cruzada.
        '''
        reshaped_x = x.reshape((x.shape[0], -1))

        if cross:
            scores = cross_val_score(self, reshaped_x, y, cv=15, scoring='f1_micro', n_jobs=-1)
            mean = np.round(scores.mean(), decimals=2)
            std = np.round(scores.std(), decimals=2)

            fig, ax = plt.subplots(figsize=(10,10))
            sns.histplot(scores, label=f'Média: {mean} -- Desvio Padrão: {std}', ax=ax)
            ax.set_xlabel(r'$F_1$ score', size=13)
            ax.set_ylabel('Histograma', size=13)
            ax.tick_params(labelsize=13)
            ax.legend()

            tax = ax.twinx()
            sns.kdeplot(scores, ax=tax)
            tax.set_ylabel('Função de densidade', size=13)

        else:
            preds = self.predict(reshaped_x)

            print(classification_report(y, preds))