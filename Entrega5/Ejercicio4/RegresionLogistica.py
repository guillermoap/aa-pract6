import pandas as pd
import math
import copy
import numpy as np
import collections
import copy
from sklearn.metrics.pairwise import euclidean_distances
import time


class RegresionLogisticaModel():

    def __init__(self):
        self.train_set = None
        self.target_attr = None
        self.attrs = None
        self.examples_quantity = None
        self.X = None
        self.y = None
        self.weights = None

    def h(self, weights, x):
        return 1 / (1 + math.exp(-np.inner(weights, x.values)))

    def weight_function(self):
        summation = sum(self.y.iloc[i] * math.log(self.h(self.weights, self.X.iloc[i]))
                        + (1 - self.y.iloc[i]) * math.log(1 - self.h(self.weights, self.X.iloc[i]))
                        for i in range(0, self.examples_quantity))
        return (-1/self.examples_quantity) * summation

    def train(self, train_set, target_attr, attrs, delta=0.0001, alpha=0.1, max_iter=10000000000000):
        print('Comienzo del entrenamiento...\n')
        start = time.time()
        self.train_set = train_set
        self.target_attr = target_attr
        self.attrs = attrs
        self.examples_quantity = train_set.shape[0]
        X = train_set.iloc[:, train_set.columns != self.target_attr]
        X.insert(loc=0, column='x0', value=np.ones(self.examples_quantity))
        self.X = X
        self.y = train_set.iloc[:, train_set.columns.get_loc(self.target_attr)]
        self.weights = np.ones(len(attrs) + 1)
        distance = 10
        iter = 1
        while distance > delta and iter < max_iter:
            print('Iteracion: ', iter)
            print('Pesos: ', self.weights)
            print('Funcion de costo: ', self.weight_function(), '\n')
            old_weights = np.copy(self.weights)
            weights = np.copy(self.weights)
            for j in range(0, len(self.weights)):
                summation = sum((self.h(old_weights, self.X.iloc[i]) - self.y.iloc[i]) * self.X.iloc[i][j] for i in range(0, self.examples_quantity))
                weights[j] = weights[j] - (alpha/self.examples_quantity * summation)
            distance = euclidean_distances(X=old_weights.reshape(1, -1), Y=weights.reshape(1, -1))
            self.weights = weights
            iter += 1

        print('--------------------------------\n')
        print('Fin del entrenamiento\n')
        print('Total de iteraciones: ', iter)
        print('Tiempo total de entrenamiento: ', time.time() - start)
        print('Pesos finales: ', self.weights)
        print('Funcion de costo: ', self.weight_function(), '\n')
        print('--------------------------------\n')

    def predict(self, dataframe):
        dataframe.insert(loc=0, column='x0', value=np.ones(dataframe.shape[0]))
        values = self.evaluate(dataframe.copy().iloc[:, dataframe.columns != self.target_attr])
        dataframe["prediction"] = values
        return dataframe[[self.target_attr, "prediction"]]

    def evaluate(self, dataframe):
        values = pd.Series([])
        for row in dataframe.iterrows():
            value = self._single_evaluate(row[1])
            values.at[row[0]] = value
        return values

    def _single_evaluate(self, series):
        return 1 if self.h(self.weights, series) >= 0.5 else 0


def calculate_error(df, target):
    result = (df[target] == df["prediction"]).value_counts()
    return result.get(False, 0)


def cross_validation(K, dataframe, target_attr, attrs, delta, alpha, max_iter):
    # Se randomiza el dataset
    dataframe = dataframe.sample(frac=1)
    # Se divide el dataset en partes
    dfsize = len(dataframe)
    # Número de grupos
    size = (dfsize // K) + 1
    data_training = []
    # Se asigna un grupo a cada columna y luego se los agrupa
    # Cada grupo se usa para validación
    for g, df in dataframe.groupby(np.arange(dfsize) // size):
        data_training.append((dataframe.copy().drop(df.index), df))

    model = RegresionLogisticaModel()

    details = []
    i = 1

    for di, ti in data_training:
        print("Training - %d / %d" % (i, K))
        model.train(di, target_attr, attrs, delta, alpha, max_iter)

        print(" -> Predicting - %d / %d" % (i, K))
        prediction = model.predict(ti)
        details.append(calculate_error(prediction, target_attr))
        print(" -> Error - %d / %d - %d out of %d" % (i, K, details[i - 1], size))
        i += 1

    error = {"error": sum(details) / float(K), "test_size": size}
    error_percent = error["error"] * 100.0 / error["test_size"]

    print("Cross validation Average Error")
    print("----------------")
    print(" * Error     : %.2f" % error["error"])
    print(" * T(i) Size : %.2f" % error["test_size"])
    print(" * Error %%   : %.2f" % error_percent)
    print("\n")

    return error


def train_predict(train, test, target_attr, attrs, delta=0.0001, alpha=0.1, max_iter=10000000000000):
    model = RegresionLogisticaModel()
    model.train(train, target_attr, attrs, delta, alpha, max_iter)
    result = model.predict(test)
    error = calculate_error(result, target_attr)
    test_len = len(test)
    error_percent = error * 100 / test_len

    print("Error on Test")
    print("-------------")
    print(" * Error         : %2.f" % calculate_error(result, target_attr))
    print(" * Size (test)   : %2.f" % len(test))
    print(" * Error %%       : %.2f" % error_percent)
    print("\n")

    return model