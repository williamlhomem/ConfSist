# importações
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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

def metrics_conversion(baseline, model, sample_x, temps, sample_y):
    '''
    Função responsável por realizar a conversão entre as métricas do modelo de ML para as métricas de negócio.
    I/O:
        baseline: um hydro_utils.models.baseline_model contendo o modelo base;
        model: um RandomForestClassifier contendo o modelo de ML;
        sample_x: um numpy array contendo a entrada do modelo de ML;
        temps: um numpy array contendo as temperaturas de entrada do baseline model;
        sample_y: um numpy array contendo os rótulos.
    '''
    # predições dos modelos
    base_preds = baseline.predict(temps)
    ml_preds = model.predict(sample_x.reshape((sample_x.shape[0], -1)))

    # criação do df de comparação
    compare_df = pd.DataFrame()
    compare_df['baseline'] = base_preds
    compare_df['model'] = ml_preds
    compare_df['actual'] = sample_y

    # determinação da métrica de negócio
    def metric(series):
        series = series.values
        true = series[0]
        pred = series[1]
        
        # caso da identificação de baixa eficiência
        if (true == 20) and (pred == 20):
            cost = -500
        # caso da identificação da falha total
        elif (true == 3) and (pred == 3):
            cost = 1500
        # caso da identificação equivocada da falha total:
        elif (true == 3) and (pred != 3):
            cost = 2500
        else:
            cost = 0
        
        return cost

    compare_df['baseline_cost'] = compare_df[['actual', 'baseline']].apply(metric, axis='columns')
    compare_df['model_cost'] = compare_df[['actual', 'model']].apply(metric, axis='columns')

    return compare_df

class pca:
    def __init__(self, data):
        '''
        Objeto responsável pela redução dimensional de um conjunto por meio de PCA.
        I/O:
            data: um pandas dataframe contendo os dados que serão reduzidos
        '''
        self.X = data.copy()

        # normalização dos dados
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.X)

    def ev_curve(self):
        '''
        Método que calcula e plota a curva de variância explicada para análise do número de componentes.
        '''
        range_comp = range(1, 100)
        ve = []
        for comp in range_comp:
            model = PCA(n_components=comp, random_state=42).fit(self.X)
            ve.append(model.explained_variance_ratio_.sum())
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(range_comp, ve)
        ax.set_xlabel('Componentes', size=15)
        ax.set_ylabel('Variância Explicada', size=15)
        ax.tick_params(labelsize=15)
        ax.grid()

    def d_reduction(self, n_components):
        '''
        Método que realiza a redução dimensional.
        I/O:
            n_components: um inteiro que indica o número de componentes usado para a redução.
        '''
        self.model = PCA(n_components=n_components, random_state=42).fit(self.X)
        return pd.DataFrame(self.model.transform(self.X))

class kmeans():
    def __init__(self, data):
        '''
        Objeto que cria um modelo kmeans.
        I/O:
            data: conjunto de dados contendo os dados que serão agrupados.
        '''
        self.X = data.copy()
    
    def cluster_criterion(self, n_clusters, verbose=False):
        '''
        Método que verifica o critério de inércia para definição no número de grupos ideal para o agrupamento.
        I/O:
            n_clusters: inteiro que indica o número máximo de clusters participantes do critério;
            verbose: booleano que indica se deve ser informado, ou não, o andamento do cálculo de inércia.
        '''
        ine = []
        n_range = range(2, n_clusters + 1)

        for i in tqdm(n_range):
            kmeans = KMeans(n_clusters=i, max_iter=5000, random_state=42)
            kmeans.fit(self.X)
            ine.append(kmeans.inertia_)            
            if verbose:
                print(f'Número de agrupamentos: {i} --- Inércia: {ine[i-2]}')
               
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(n_range, ine)
        ax.set_xticks(n_range)
        ax.set_xlabel("Número de agrupamentos")
        ax.set_ylabel("Inércia")
        ax.grid()
        
    def create_model(self, n_clusters):
        '''
        Método que cria o modelo kmeans.
        I/O:
            n_cluster: inteiro que indica o número de grupos existentes no conjunto de dados.
        '''
        self.model = KMeans(n_clusters=n_clusters, max_iter=5000, random_state=42)
        self.model.fit(self.X)
    
    def clustering(self):
        '''
        Método que realiza o clustering do conjunto de dados trabalhado.
        I/O:
            return cluster_data: um pandas dataframe contendo o conjunto de dados com o rótulo dos agrupamentos feitos pelo modelo.
        ''' 
        self.Y_ = self.model.predict(self.X)
        cluster_data = self.X.copy()
        cluster_data['cluster'] = self.Y_
        
        return cluster_data