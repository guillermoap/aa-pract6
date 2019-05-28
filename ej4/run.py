import warnings
import datetime
import pandas as pd
import seaborn as sn
from time import time
from sklearn import metrics
from sklearn.metrics import pairwise_distances, adjusted_rand_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from src.classifier import Classifier
import matplotlib.pyplot as plt
import numpy as np
from pandas_ml import ConfusionMatrix

warnings.filterwarnings('ignore')
party_map = {
    'Frente Amplio': 0,
    'Partido Nacional': 1,
    'Partido Colorado': 2,
    'La Alternativa': 3,
    'Unidad Popular': 4,
    'Partido de la Gente': 5,
    'PERI': 6,
    'Partido de los Trabajadores': 7,
    'Partido Digital': 8,
    'Partido Verde': 9,
    'Partido de Todos': 10
}

candidatos = {
    '1': 0,
    '2': 0,
    '3': 0,
    '4': 0,
    '5': 1,
    '6': 1,
    '8': 1,
    '9': 1,
    '10': 1,
    '11': 1,
    '12': 2,
    '13': 2,
    '14': 2,
    '15': 2,
    '16': 2,
    '17': 2,
    '18': 3,
    '19': 4,
    '20': 5,
    '21': 6,
    '22': 7,
    '23': 8,
    '24': 9,
    '25': 10
}

def load_data():
    # Importamos los datos utilizando pandas
    data=pd.read_csv("./data.csv")

    # Creo la tabla de candidatos a mano
    candidatos=pd.DataFrame(
        [
            [1,'Oscar Andrade', 'Frente Amplio', 0],
            [2,'Mario Bergara', 'Frente Amplio', 0],
            [3,'Carolina Cosse', 'Frente Amplio', 0],
            [4,'Daniel Martínez', 'Frente Amplio', 0],
            [5,'Verónica Alonso', 'Partido Nacional', 1],
            [6,'Enrique Antía', 'Partido Nacional', 1],
            [8,'Carlos Iafigliola', 'Partido Nacional', 1],
            [9,'Luis Lacalle Pou', 'Partido Nacional', 1],
            [10,'Jorge Larrañaga', 'Partido Nacional', 1],
            [11,'Juan Sartori', 'Partido Nacional', 1],
            [12,'José Amorín', 'Partido Colorado', 2],
            [13,'Pedro Etchegaray', 'Partido Colorado', 2],
            [14,'Edgardo Martínez', 'Partido Colorado', 2],
            [15,'Héctor Rovira', 'Partido Colorado', 2],
            [16,'Julio María Sanguinetti', 'Partido Colorado', 2],
            [17,'Ernesto Talvi', 'Partido Colorado', 2],
            [18,'Pablo Mieres', 'La Alternativa', 3],
            [19,'Gonzalo Abella', 'Unidad Popular', 4],
            [20,'Edgardo Novick', 'Partido de la Gente', 5],
            [21,'Cèsar Vega', 'PERI', 6],
            [22,'Rafael Fernández', 'Partido de los Trabajadores', 7],
            [23,'Justin Graside', 'Partido Digital', 8],
            [24,'Gustavo Salle', 'Partido Verde', 9],
            [25,'Carlos Techera', 'Partido de Todos', 10]
        ],
        columns=['candidatoId','name','party','idPartido'],
    )

    data=data.merge(candidatos,on=['candidatoId'])

    # Sólo por si necesita, cargamos un diccionario con el texto de cada pregunta
    preguntas={
        '1': 'Controlar la inflación es más importante que controlar el desempleo. ',
        '2': 'Hay que reducir la cantidad de funcionarios pùblicos',
        '3': 'Deberia aumentar la carga de impuestos para los ricos.',
        '4': 'El gobierno no debe proteger la industria nacional, si las fábricas no son competitivas esta bien que desaparezcan.',
        '5': 'La ley de inclusión financiera es positiva para la sociedad. ',
        '6': 'Algunos sindicatos tienen demasiado poder. ',
        '7': 'Cuanto más libre es el mercado, más libre es la gente. ',
        '8': 'El campo es y debe ser el motor productivo de Uruguay. ',
        '9': 'La inversión extranjera es vital para que Uruguay alcance el desarrollo. ',
        '10': 'Los supermercados abusan del pueblo con sus precios excesivos. ',
        '11': 'Con la vigilancia gubernamental (escuchas telefonicas, e-mails y camaras de seguridad) el que no tiene nada que esconder, no tiene de que preocuparse. ',
        '12': 'La pena de muerte debería ser una opción para los crímenes mas serios. ',
        '13': 'Uruguay debería aprobar más leyes anti corrupción y ser más duro con los culpables. ',
        '14': 'Las FF.AA. deberían tener un rol activo en la seguridad pública. ',
        '15': 'Las carceles deberían ser administradas por organizaciones privadas. ',
        '16': 'Hay que aumentar el salario de los policias significativamente. ',
        '17': 'Para los delitos más graves hay que bajar la edad de imputabilidad a 16 años. ',
        '18': 'Uruguay no necesita un ejército. ',
        '19': 'Uruguay es demasiado generoso con los inmigrantes. ',
        '20': 'La ley trans fue un error. ',
        '21': 'El feminismo moderno no busca la igualdad sino el poder. ',
        '22': 'Para la ley no deberia diferenciarse homicidio de femicidio. ',
        '23': 'La separación de estado y religión me parece importante. ',
        '24': 'La legalización de la marihuana fue un error. ',
        '25': 'La legalización del aborto fue un error. ',
        '26': 'El foco del próximo gobierno debe ser mejorar la educación pública. '
    }

    # Ordeno los datos por partido y luego por candidato
    data.drop('fecha', axis=1, inplace=True)
    data = data.sort_values(by=['party','name'])
    parties = []
    parties_count  = dict(data.groupby('party').party.count())
    for row in parties_count.items():
        if row[1] > 1000:
            parties.append(row[0])
    data = data.query(f'party in {parties}')
    data.drop(['name', 'party'], axis=1, inplace=True)
    # Para PCA solo usamos las preguntas
    candidatos = data.iloc[:,1:28]
    partidos = data.iloc[:,2:29]
    return candidatos, partidos, data

def output_results(title, actual, predicted):
    print(f'----------------------------------------')
    print(f'######### {title} #########')
    print('score: micro, macro')
    print(f"precision: {precision_score(actual, predicted, average='micro')}, {precision_score(actual, predicted, average='macro')}")
    print(f"recall: {recall_score(actual, predicted, average='micro')}, {recall_score(actual, predicted, average='macro')}")
    print(f"f1: {f1_score(actual, predicted, average='micro')}, {f1_score(actual, predicted, average='macro')}")
    print(f'----------------------------------------')

def plot_confusion_matrix(cm, title='Confusion matrix', candidato=True):
    plt.title(title)
    if candidato:
        df_cm = pd.DataFrame(cm, index = [i for i in candidatos], columns = [i for i in candidatos])
    plt.figure()
    sn.heatmap(df_cm, annot=True)

def candidato_mapper(data, target):
    data = list(map(lambda x: candidatos[f'{x}'], data))
    target = list(map(lambda x: candidatos[f'{x}'], target))
    return data, target

def run():
    start_time = time()
    data_cand, data_part, full_data = load_data()
    # numeric_parties  = full_data.party.map(party_map)
    train_c, test_c = train_test_split(data_cand, test_size=0.2)
    train_p, test_p = train_test_split(data_part, test_size=0.2)
    candidatos_clf = Classifier(train_c.drop('candidatoId', axis=1), train_c.candidatoId)
    partidos_clf = Classifier(train_p.drop('idPartido', axis=1), train_p.idPartido)

    cand_solver = candidatos_clf._predict()
    n_cand, pca_cand_solver = candidatos_clf._pca()
    part_solver = partidos_clf._predict()
    n_part, pca_part_solver = partidos_clf._pca()

    cand_pred = candidatos_clf.classify(test_c.drop('candidatoId', axis=1), test_c.candidatoId, cand_solver)
    pca_cand_pred = candidatos_clf.classify(test_c.drop('candidatoId', axis=1), test_c.candidatoId, pca_cand_solver, n_cand)
    part_pred = partidos_clf.classify(test_p.drop('idPartido', axis=1), test_p.idPartido, part_solver)
    pca_part_pred = partidos_clf.classify(test_p.drop('idPartido', axis=1), test_p.idPartido, pca_part_solver, n_part)

    output_results(f'CANDIDATOS | {cand_solver}', test_c.candidatoId, cand_pred)
    output_results(f'CANDIDATOS_PCA | {pca_cand_solver}, {n_cand}', test_c.candidatoId, pca_cand_pred)
    output_results(f'PARTIDOS | {part_solver}', test_p.idPartido, part_pred)
    output_results(f'PARTIDOS_PCA | {pca_part_solver}, {n_part}', test_p.idPartido, pca_part_pred)
    cand_part_target, cand_part_pred = candidato_mapper(test_c.candidatoId, cand_pred)
    output_results(f'PARTIDO | {cand_solver}', cand_part_target, cand_part_pred)

    # cm_cand = confusion_matrix(test_c.candidatoId, cand_pred)
    # cm_pca_cand = confusion_matrix(test_c.candidatoId, pca_cand_pred)
    # cm_part = confusion_matrix(test_p.idPartido, part_pred)
    # cm_pca_part = confusion_matrix(test_p.idPartido, pca_part_pred)
    # plot_confusion_matrix(cm_cand, 'CANDIDATOS')
    elapsed_time = time() - start_time
    print(f'----------------------------------------')
    print(f'TOTAL TIME: {datetime.timedelta(seconds=elapsed_time)}')

if __name__ == "__main__":
    run()
