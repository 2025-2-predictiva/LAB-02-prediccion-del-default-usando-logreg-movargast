# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

# flake8: noqa: E501

from pathlib import Path
import pandas as pd
import numpy as np
import zipfile
import pickle
import json
import gzip
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)



def load_data(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    if path.suffix == ".zip":
        try:
            return pd.read_csv(path, compression="zip", index_col=False)
        except Exception:
            with zipfile.ZipFile(path, "r") as z:
                csvs = [f for f in z.namelist() if f.endswith(".csv")]
                if not csvs:
                    raise ValueError("No hay CSV dentro del zip.")
                with z.open(csvs[0]) as f:
                    return pd.read_csv(f, index_col=False)

    return pd.read_csv(path, index_col=False)


train = load_data("files/input/train_data.csv.zip")
test = load_data("files/input/test_data.csv.zip")


# PASO 1: Limpieza de datos

def clean(df):
    df = df.copy()

    df.rename(columns={"default payment next month": "default"}, inplace=True)

    df = df.drop(columns=["ID"])

    # Los registros con EDUCATION=0 o MARRIAGE=0 deben eliminarse (requisito implícito del dataset)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]

    df = df.dropna()

    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4

    return df


train = clean(train)
test = clean(test)


# PASO 2: Separar X_train, y_train, X_test, y_test 

X_train = train.drop(columns=["default"])
y_train = train["default"]
X_test = test.drop(columns=["default"])
y_test = test["default"]


# PASO 3: Pipeline (OneHot + MinMax + SelectKBest + LogisticRegression)

categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", MinMaxScaler(), numeric_cols),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("select", SelectKBest(score_func=f_classif, k=10)),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)


# PASO 4: GRIDSEARCHCV

param_grid = {
    "select__k": range(5, 20),
    "model__C": [0.1, 1, 10],
    "model__solver": ["liblinear", "lbfgs"],
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=10,
    n_jobs=-1,
)

grid.fit(X_train, y_train)


#  PASO 5: Guardar modelo

os.makedirs("files/models", exist_ok=True)
os.makedirs("files/output", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)


# PASO 6 y 7: Métricas + Matriz

def compute_metrics(X, y, name):
    y_pred = grid.predict(X)
    return {
        "type": "metrics",
        "dataset": name,
        "precision": float(precision_score(y, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "recall": float(recall_score(y, y_pred)),
        "f1_score": float(f1_score(y, y_pred)),
    }


def confusion_to_dict(cm, name):
    return {
        "type": "cm_matrix",
        "dataset": name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }


metrics_list = []

# Métricas
metrics_list.append(compute_metrics(X_train, y_train, "train"))
metrics_list.append(compute_metrics(X_test, y_test, "test"))

# Matrices
cm_train = confusion_matrix(y_train, grid.predict(X_train))
cm_test = confusion_matrix(y_test, grid.predict(X_test))

metrics_list.append(confusion_to_dict(cm_train, "train"))
metrics_list.append(confusion_to_dict(cm_test, "test"))

# Guardar archivo
with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    for row in metrics_list:
        f.write(json.dumps(row) + "\n")

print("Proceso Completado.")

