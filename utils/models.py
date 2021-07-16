from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class RandForestClass(RandomForestClassifier):
    '''
    Objeto responsável pela criação de um modelo de Floresta de Decisão Aleatória. Atributos e métodos herdados de sklearn.ensemble.RandomForestClassifier.
    '''
    def report(self, x, y):
        '''
        Função responsável por informar as métricas de performance relacionadas ao modelo de classificação.
        I/O:
            x: um pandas dataframe ou um numpy array contendo as *features* do conjunto de dados;
            y: um pandas series ou um numpy array contendo os rótulos do conjunto de dados.
        '''
        preds = self.predict(x)
        print(classification_report(y, preds))