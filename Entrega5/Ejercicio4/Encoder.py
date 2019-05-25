import pandas as pd
import math
import collections
import time
from functools import reduce
from scipy.stats import norm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings


class MLTransformationModel:

    def fit(self, dataframe):
        pass

    def fit_transform(self, dataframe):
        self.fit(dataframe)
        return self.transform(dataframe)

    def transform(self, dataframe):
        pass


class Encoder(MLTransformationModel):
    def __init__(self, attrs=None, target_attr='Class/ASD'):
        self.attrs = attrs
        self.target_attr = target_attr
        self.encoders = {}

    def fit(self, dataframe):
        # Creo lista de columnas
        self.attrs = self.attrs or list(dataframe)

        # Creo los label encoders para cada elemento
        for attr in self.attrs:
            uniq_vals = list(dataframe[attr].unique())
            # print uniq_vals
            warnings.filterwarnings(action='ignore', category=DeprecationWarning)
            self.encoders[attr] = LabelEncoder()
            self.encoders[attr].fit(uniq_vals)
            print("processing %s ... " % attr)
        return self

    def transform(self, dataframe):
        df = dataframe.copy()
        ## aplico los label encoders que se prepararon
        for attr in self.attrs:
            print("applying label encoder for %s" % attr)
            df[attr] = df[attr].apply(lambda x: self.encoders[attr].transform([x])[0])

        ## OneHotEncoder - Part
        print("preparing OneHotEconder")
        oc_attrs = self.attrs.copy()
        if self.target_attr in oc_attrs:
            oc_attrs.remove(self.target_attr)

        oc = OneHotEncoder(dtype=int)
        oc.fit(df[oc_attrs])

        # Genero la lista de columnas para agregar en el dataset (están en orden)
        print("Creating new columns for encoded data")
        columns = []
        for attr in oc_attrs:
            values = df[attr].unique()
            for val in sorted(values):
                label = self.encoders[attr].inverse_transform(val)
                # label = label.strip()
                ncol = "{0}_{1}".format(attr, label)
                columns.append(ncol)

        # Returns a matrix, for simplicity put everything in Pandas to drop unwanted columns
        print("transforming data using one hot encoder")
        matrix_data = oc.transform(df[oc_attrs])
        pdX = pd.DataFrame(matrix_data.todense(), columns=columns)
        for attr in oc_attrs:
            df.drop(attr, 1, inplace=True)

        print("merging dataframe")
        final_df = pd.merge(df, pdX, left_index=True, right_index=True)

        # Convert Pandas DataFrame or Series to numpy.array

        # Finally: Generate the matrix X, clean with all the attributes I need.
        # X = final_df.as_matrix()
        return final_df