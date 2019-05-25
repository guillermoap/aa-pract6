
# coding: utf-8

import pandas as pd
import RegresionLogistica as rl
import ColumnModifiers as cm
from arffConvert import convertSingleFileToCSV
import Encoder as e
import numpy as np


convertSingleFileToCSV("Autism-Adult-Data.arff")
dataframe = pd.read_csv("Autism-Adult-Data.csv", sep=",")


dataframe.head()

for column in dataframe.columns:
    print("Columna %s - %s \n" % (column, dataframe[column].unique()))

# Se definen los column modifiers

columns_modifiers_unknown_as_most_common = {
    "A1_Score": cm.MostCommonValue(),
    "A2_Score": cm.MostCommonValue(),
    "A3_Score": cm.MostCommonValue(),
    "A4_Score": cm.MostCommonValue(),
    "A5_Score": cm.MostCommonValue(),
    "A6_Score": cm.MostCommonValue(),
    "A7_Score": cm.MostCommonValue(),
    "A8_Score": cm.MostCommonValue(),
    "A9_Score": cm.MostCommonValue(),
    "A10_Score": cm.MostCommonValue(),
    "age": cm.Discretization(),
    "gender": cm.MostCommonValue(),
    "ethnicity": cm.MostCommonValue(), # Tiene vacios
    "jundice": cm.MostCommonValue(),
    "austim": cm.MostCommonValue(),
    "contry_of_res": cm.MostCommonValue(),
    "used_app_before": cm.MostCommonValue(),
    # "result": cm.Discretization(), Removido
    # "age_desc": cm.Discretization(), Removido, siempre tiene el mismo valor
    "relation": cm.MostCommonValue() #,  Tiene Vacios
    # "Class/ASD": cm.Discretization(), Target column
}

columns_modifiers_unknown_as_most_common_no_discretization = {
    "A1_Score": cm.MostCommonValue(),
    "A2_Score": cm.MostCommonValue(),
    "A3_Score": cm.MostCommonValue(),
    "A4_Score": cm.MostCommonValue(),
    "A5_Score": cm.MostCommonValue(),
    "A6_Score": cm.MostCommonValue(),
    "A7_Score": cm.MostCommonValue(),
    "A8_Score": cm.MostCommonValue(),
    "A9_Score": cm.MostCommonValue(),
    "A10_Score": cm.MostCommonValue(),
    "age": cm.MostCommonValue(),
    "gender": cm.MostCommonValue(),
    "ethnicity": cm.MostCommonValue(), # Tiene vacios
    "jundice": cm.MostCommonValue(),
    "austim": cm.MostCommonValue(),
    "contry_of_res": cm.MostCommonValue(),
    "used_app_before": cm.MostCommonValue(),
    # "result": cm.Discretization(), Removido
    # "age_desc": cm.Discretization(), Removido, siempre tiene el mismo valor
    "relation": cm.MostCommonValue() #,  Tiene Vacios
    # "Class/ASD": cm.Discretization(), Target column
}

columns_modifiers_unknown_as_most_common_equal_calification = {
    "A1_Score": cm.MostCommonValueEqualCalification(),
    "A2_Score": cm.MostCommonValueEqualCalification(),
    "A3_Score": cm.MostCommonValueEqualCalification(),
    "A4_Score": cm.MostCommonValueEqualCalification(),
    "A5_Score": cm.MostCommonValueEqualCalification(),
    "A6_Score": cm.MostCommonValueEqualCalification(),
    "A7_Score": cm.MostCommonValueEqualCalification(),
    "A8_Score": cm.MostCommonValueEqualCalification(),
    "A9_Score": cm.MostCommonValueEqualCalification(),
    "A10_Score": cm.MostCommonValueEqualCalification(),
    "age": cm.Discretization(),
    "gender": cm.MostCommonValueEqualCalification(),
    "ethnicity": cm.MostCommonValueEqualCalification(), # Tiene vacios
    "jundice": cm.MostCommonValueEqualCalification(),
    "austim": cm.MostCommonValueEqualCalification(),
    "contry_of_res": cm.MostCommonValueEqualCalification(),
    "used_app_before": cm.MostCommonValueEqualCalification(),
    # "result": cm.Discretization(), Removido
    # "age_desc": cm.Discretization(), Removido, siempre tiene el mismo valor
    "relation": cm.MostCommonValueEqualCalification() #,  Tiene Vacios
    # "Class/ASD": cm.Discretization(), Target column
}

columns_modifiers_unknown_as_most_common_equal_calification_no_discretization = {
    "A1_Score": cm.MostCommonValueEqualCalification(),
    "A2_Score": cm.MostCommonValueEqualCalification(),
    "A3_Score": cm.MostCommonValueEqualCalification(),
    "A4_Score": cm.MostCommonValueEqualCalification(),
    "A5_Score": cm.MostCommonValueEqualCalification(),
    "A6_Score": cm.MostCommonValueEqualCalification(),
    "A7_Score": cm.MostCommonValueEqualCalification(),
    "A8_Score": cm.MostCommonValueEqualCalification(),
    "A9_Score": cm.MostCommonValueEqualCalification(),
    "A10_Score": cm.MostCommonValueEqualCalification(),
    "age": cm.MostCommonValueEqualCalification(),
    "gender": cm.MostCommonValueEqualCalification(),
    "ethnicity": cm.MostCommonValueEqualCalification(), # Tiene vacios
    "jundice": cm.MostCommonValueEqualCalification(),
    "austim": cm.MostCommonValueEqualCalification(),
    "contry_of_res": cm.MostCommonValueEqualCalification(),
    "used_app_before": cm.MostCommonValueEqualCalification(),
    # "result": cm.Discretization(), Removido
    # "age_desc": cm.Discretization(), Removido, siempre tiene el mismo valor
    "relation": cm.MostCommonValueEqualCalification() #,  Tiene Vacios
    # "Class/ASD": cm.Discretization(), Target column
}

# Se define target_attr

target_attr = 'Class/ASD'

# Se aplican column modifiers y one hot encoding al dataframe para cada caso

encoder = e.Encoder()
dataframe_unknown_as_most_common = encoder.fit_transform(cm.apply_column_modifiers(dataframe.copy(), columns_modifiers_unknown_as_most_common))
dataframe_unknown_as_most_common_no_discretization = encoder.fit_transform(cm.apply_column_modifiers(dataframe.copy(), columns_modifiers_unknown_as_most_common_no_discretization))
dataframe_unknown_as_most_common_equal_calification = encoder.fit_transform(cm.apply_column_modifiers(dataframe.copy(), columns_modifiers_unknown_as_most_common_equal_calification))
dataframe_unknown_as_most_common_equal_calification_no_discretization = encoder.fit_transform(cm.apply_column_modifiers(dataframe.copy(), columns_modifiers_unknown_as_most_common_equal_calification_no_discretization))

# Se definen los atributos (sin considerar target_attr) para cada caso

attrs_unknown_as_most_common = np.delete(dataframe_unknown_as_most_common.columns.values, 0)
attrs_unknown_as_most_common_no_discretization = np.delete(dataframe_unknown_as_most_common_no_discretization.columns.values, 0)
attrs_unknown_as_most_common_equal_calification = np.delete(dataframe_unknown_as_most_common_equal_calification.columns.values, 0)
attrs_unknown_as_most_common_equal_calification_no_discretization = np.delete(dataframe_unknown_as_most_common_equal_calification_no_discretization.columns.values, 0)

# Se definen el set de train y el de test para cada caso

training_percent = .4/.5

train_unknown_as_most_common = dataframe_unknown_as_most_common.sample(frac=training_percent).copy()
test_unknown_as_most_common = dataframe_unknown_as_most_common.drop(train_unknown_as_most_common.index).copy()

train_unknown_as_most_common_no_discretization = dataframe_unknown_as_most_common_no_discretization.sample(frac=training_percent).copy()
test_unknown_as_most_common_no_discretization = dataframe_unknown_as_most_common_no_discretization.drop(train_unknown_as_most_common_no_discretization.index).copy()

train_unknown_as_most_common_equal_calification = dataframe_unknown_as_most_common_equal_calification.sample(frac=training_percent).copy()
test_unknown_as_most_common_equal_calification = dataframe_unknown_as_most_common_equal_calification.drop(train_unknown_as_most_common.index).copy()

train_unknown_as_most_common_equal_calification_no_discretization = dataframe_unknown_as_most_common_equal_calification_no_discretization.sample(frac=training_percent).copy()
test_unknown_as_most_common_equal_calification_no_discretization = dataframe_unknown_as_most_common_equal_calification_no_discretization.drop(train_unknown_as_most_common_equal_calification_no_discretization.index).copy()


# Cross validation con 4/5

k = 10
print("\nCross validation - Unknown as most common - Discretization")
rl.cross_validation(k, train_unknown_as_most_common, target_attr, attrs=attrs_unknown_as_most_common, alpha=1, delta=0.01, max_iter=5)
print("\nCross validation - Unknown as most common - No Discretization")
rl.cross_validation(k, train_unknown_as_most_common_no_discretization, target_attr, attrs=attrs_unknown_as_most_common_no_discretization, alpha=1, delta=0.01, max_iter=5)
print("\nCross validation - Unknown as most common equal calification - Discretization")
rl.cross_validation(k, train_unknown_as_most_common_equal_calification, target_attr, attrs=attrs_unknown_as_most_common_equal_calification, alpha=1, delta=0.01, max_iter=5)
print("\nCross validation - Unknown as most common equal calification - No Discretization")
rl.cross_validation(k, train_unknown_as_most_common_equal_calification_no_discretization, target_attr, attrs=attrs_unknown_as_most_common_equal_calification_no_discretization, alpha=1, delta=0.01, max_iter=5)


# Se entrena con 4/5 y prueba con 1/5


print("\nExecution - Unknown as most common - Discretization")
missing_as_most_common_model = rl.train_predict(train_unknown_as_most_common.sample(frac=1), test_unknown_as_most_common.sample(frac=1), target_attr, attrs=attrs_unknown_as_most_common, alpha=1, delta=0.01, max_iter=5)

print("\nExecution - Unknown as most common - No Discretization")
missing_as_most_common_no_disc_model = rl.train_predict(train_unknown_as_most_common_no_discretization.sample(frac=1), test_unknown_as_most_common.sample(frac=1), target_attr, attrs=attrs_unknown_as_most_common_no_discretization, alpha=1, delta=0.01, max_iter=5)

print("\nExecution - Unknown as most common equal calification - Discretization")
missing_as_most_common_equal_calification_model = rl.train_predict(train_unknown_as_most_common_equal_calification.sample(frac=1), test_unknown_as_most_common_equal_calification.sample(frac=1), target_attr, attrs=attrs_unknown_as_most_common_equal_calification, alpha=1, delta=0.01, max_iter=5)

print("\nExecution - Unknown as most common equal calification - No Discretization")
missing_as_most_common_equal_calification_no_disc_model = rl.train_predict(train_unknown_as_most_common_equal_calification_no_discretization.copy().sample(frac=1), test_unknown_as_most_common_equal_calification_no_discretization.copy().sample(frac=1), target_attr, attrs=attrs_unknown_as_most_common_equal_calification_no_discretization, alpha=1, delta=0.01, max_iter=5)


