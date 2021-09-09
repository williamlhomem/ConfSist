import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler

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

class AE:
    def __init__(self, x):
        self.x = x.copy()

        # normalização dos dados
        self.scaler = StandardScaler()
        self.x[self.x.columns] = self.scaler.fit_transform(self.x[self.x.columns])

    def create_model(self, lr=0.1):
        input_shape = self.x.shape[1]

        # criação do modelo
        ## encoder
        input_layer = tf.keras.layers.Input(shape=(input_shape,), name='input_layer')
        x = tf.keras.layers.Dense(units=32, activation='relu')(input_layer)
        x = tf.keras.layers.Dense(units=16, activation='relu')(x)

        ## espaço latente
        z = tf.keras.layers.Dense(units=8, activation='relu', name='latent_layer')(x)

        ## decoder
        x = tf.keras.layers.Dense(units=16, activation='relu')(z)
        x = tf.keras.layers.Dense(units=32, activation='relu')(x)
        output_layer = tf.keras.layers.Dense(units=input_shape, activation='relu', name='output_layer')(x)

        # compilação do modelo
        self.model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    def train_model(self, batch_size=32, epochs=30):
        # treinamento
        hist = self.model.fit(self.x, self.x, batch_size=batch_size, epochs=epochs, validation_split=.2, verbose=0)

        # curvas de aprendizagem
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(hist.history['loss'], label='Custo de treinamento')
        ax.plot(hist.history['val_loss'], label='Custo de validação')
        
        ax.set_xlabel('Épocas', size=12)
        ax.set_ylabel('Custo', size=12)
        ax.legend()
        ax.grid()

    def classifier(self, data, lim):
        # normalização dos dados
        x = data.copy()
        x[x.columns] = self.scaler.transform(x[x.columns])

        # predições
        preds = self.model.predict(x)
        classification = []
        for y, y_ in zip(x.values, preds):
            err = mean_squared_error(y, y_)
            if err >= lim:
                classification.append(1)
            else:
                classification.append(0)
        classification = np.array(classification)

        return classification