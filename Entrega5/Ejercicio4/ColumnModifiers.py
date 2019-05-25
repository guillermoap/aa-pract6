import pandas as pd
import math
import copy
import numpy as np
import collections
import copy


def read_csv(path, sep=";"):
    return pd.read_csv(path, sep=sep)


def apply_column_modifiers(df, column_modifiers):
    for column, modifier in column_modifiers.items():
        modifier.apply(df, column)
    return df


class ColumnFunction:
    def apply(self, df, attr):
        pass


# Se deja a la columna como está
class Identity(ColumnFunction):
    def apply(self, df, attr):
        # Se remueven comillas simples
        df[attr] = df[attr].apply(lambda x: x.replace("'", "") if isinstance(x, str) else x)


class MostCommonValue(ColumnFunction):

    def __init__(self, replace_value='?'):
        self.replace_value = replace_value
        self.most_common = None

    def apply(self, df, attr):
        # Se remueven comillas simples
        df[attr] = df[attr].apply(lambda x: x.replace("'", "") if isinstance(x, str) else x)
        # Se calcula el valor más común para el atributo attr
        if self.most_common is None:
            # Para cada fila de df se reemplazan los valores faltantes en la columna attr por None (lo hace la funcion find_unknown)
            df[attr] = df[attr].apply(lambda x: self.find_unknown(x))
            # Se cuenta la cantidad de veces que se repite cada posible valor del atributo attr
            vcounts = df[attr].value_counts()
            # Se guarda el valor más común del atributo
            self.most_common = vcounts.idxmax()

        df[attr] = df[attr].apply(lambda x: self.replace_unknown(x))

    def find_unknown(self, x):
        if x == self.replace_value:
            return None
        else:
            return x

    def replace_unknown(self, x):
        if x is None:
            return self.most_common
        else:
            return x


class MostCommonValueEqualCalification():

    def __init__(self, replace_value='?'):
        self.replace_value = replace_value
        self.most_common_YES = None
        self.most_common_NO = None

    def apply(self, df, attr, target_attr='Class/ASD'):

        # Para cada fila de df se reemplazan los valores faltantes en la columna attr por None (lo hace la funcion find_unknown)
        df[attr] = df.apply(lambda x: self.find_unknown(x[attr]), axis=1)

        # Calculo el atributo más común para el atributo attr para los ejemplos positivos
        if self.most_common_YES is None:
            # Me quedo con los ejemplos que clasifican positivo
            positive_examples = df[df[target_attr] == 'YES']
            # Se cuentan la cantidad de veces que se repite cada posible valor del atributo attr entre los ejemplos positivos
            vcounts = positive_examples[attr].value_counts()
            # Se guarda el valor más común del atributo para los ejemplos positivos
            self.most_common_YES = vcounts.idxmax()

        # Calulo el atributo más común para el atributo attr para los ejemplos negativos
        if self.most_common_NO is None:
            # Me quedo con los ejemplos que clasifican negativo
            negative_examples = df[df[target_attr] == 'NO']
            # Se cuentan la cantidad de veces que se repite cada posible valor del atributo attr entre los ejemplo negativos
            vcounts = negative_examples[attr].value_counts()
            # Se guarda el valor más común del atributo para los ejemplos negativos
            self.most_common_NO = vcounts.idxmax()

        # Reemplazo todos los None en la columna attr por el valor más común entre los que clasifican igual
        df[attr] = df.apply(lambda x: self.replace_unknown(x[attr], x[target_attr]), axis=1)

    def find_unknown(self, x):
        if x == self.replace_value:
            return None
        else:
            return x

    def replace_unknown(self, x, clasification):
        if x is None:
            if clasification == 'YES':
                return self.most_common_YES
            else:
                return self.most_common_NO
        else:
            return x


class Discretization(ColumnFunction):
    def __init__(self, target_attr='Class/ASD'):
        self.target_attr = target_attr
        self.sorted_ranges = None

    def apply(self, df, attr):

        # remuevo unknowns por el valor más común antes de calcular los rangos
        MostCommonValue().apply(df, attr)
        # convierto todo a floats
        df[attr] = df[attr].apply(lambda x: float(x))

        if self.sorted_ranges == None:
            # guardo los sorted ranges para que depsues se usen en el predictor
            self.sorted_ranges = self.__ranges_attr_most_common(df, attr, self.target_attr)
        sorted_ranges = self.sorted_ranges
        # print sorted_ranges
        df[attr] = df[attr].apply(lambda x: self.__classify_by_ranges(x, sorted_ranges))

    def __ranges_attr_most_common(self, df, attr, target_attr):
        # realizo el group_by de la columna con having el valor mas comun del atributo objetivo
        dict_attr_most_commont_target = \
        df[[attr, target_attr]].sort_values(by=attr).groupby([attr]).agg(lambda x: x.value_counts().index[0]).to_dict()[
            target_attr]
        # ordeno la lista por valor en attr.
        sorts__attr_most_commont_target = collections.OrderedDict(sorted(dict_attr_most_commont_target.items()))
        boundry_limits = {'max': None, 'min': None, 'target': None}
        ranges = []
        # se recorre la lista y se compara solo en un cambio de valor en el atributo target.
        # como esta ordenada nos aseguramos siempre de tomar el mayor de los valores del rango.
        for item in sorts__attr_most_commont_target:
            if boundry_limits['target'] == None:
                boundry_limits['max'] = item
                boundry_limits['min'] = item
                boundry_limits['target'] = sorts__attr_most_commont_target[item]
            else:
                if boundry_limits['target'] != sorts__attr_most_commont_target[item]:
                    ranges.append(boundry_limits.copy())
                    boundry_limits.update({'max': item, 'min': item, 'target': sorts__attr_most_commont_target[item]})
                else:
                    boundry_limits['max'] = item
        if not ranges:
            ranges.append(boundry_limits)
        else:
            if (ranges[len(ranges) - 1]['max'] < boundry_limits['max']) and (
                    ranges[len(ranges) - 1]['target'] == boundry_limits['target']):
                ranges[len(ranges) - 1]['max'] = boundry_limits['max']
            else:
                ranges.append(boundry_limits)
        return sorted(ranges, key=lambda k: k['min'])

    def __classify_by_ranges(self, value, sorted_ranges):
        """
        Clasifica los rangos de forma continua.
        La heuristica es la siguiente:
            *Si un valor cae en un intervalo del rango, clasifica como el intervalo.
            *Si cae fuera de todos los rangos:
                *Si cae en el conjunto de todos los rangos:
                    *Clasifica con el rango mas cercano
                *Si cae fuera del conjunto de todos los rangos:
                    *Clasifica con el rango de los limites
        """
        ant_rng = sorted_ranges[0]
        ult_rng = sorted_ranges[len(sorted_ranges) - 1]
        if value < ant_rng['min']:
            return '(%s,%s)' % (ant_rng['min'], ant_rng['max'])
        if value > ult_rng['max']:
            return '(%s,%s)' % (ult_rng['min'], ult_rng['max'])
        for rng in sorted_ranges:
            if value > rng['max']:
                continue
            else:
                if value >= rng['min']:
                    return '(%s,%s)' % (rng['min'], rng['max'])
                else:
                    if abs(ant_rng['max'] - value) < abs(rng['max'] - value):
                        return '(%s,%s)' % (ant_rng['min'], ant_rng['max'])
                    else:
                        return '(%s,%s)' % (rng['min'], rng['max'])