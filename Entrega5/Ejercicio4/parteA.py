import pandas as pd
import RegresionLogistica as rl
import numpy as np
import Encoder as encoder


def get_data():
    d = {'dedicacion': ['alta', 'baja', 'media', 'media', 'alta'],
         'dificultad': ['alta', 'media', 'alta', 'alta', 'baja'],
         'horario': ['nocturno', 'matutino', 'nocturno', 'matutino', 'nocturno'],
         'humedad': ['media', 'alta', 'media', 'alta', 'media'],
         'humordoc': ['bueno', 'malo', 'malo', 'bueno', 'bueno'],
         'salva': ['si', 'no', 'si', 'no', 'si']}
    return pd.DataFrame(data=d)


def main():
    training_set = get_data()
    e = encoder.Encoder(target_attr='salva')
    encoded_training_set = e.fit_transform(training_set)

    training_percent = .4 / .5
    train = encoded_training_set.sample(frac=training_percent).copy()
    test = encoded_training_set.drop(train.index).copy()
    print('train: \n', train, '\n')
    print('test: \n', test, '\n')

    attrs = encoded_training_set.columns.values
    attrs = np.delete(attrs, 0)
    print('attrs:', attrs)

    rl.train_predict(train=train, test=test, target_attr='salva', attrs=attrs, delta=0.0001, alpha=0.1)


if __name__ == "__main__":
    main()
