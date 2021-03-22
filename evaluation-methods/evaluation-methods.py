#!/usr/bin/env python
# coding: utf-8

# # Parcial 2 Aprendizaje Automático y Análisis de Datos
# - Autor: Josue Peña Atencio - 8935601
# - Fecha: Abril 18 2020

# In[1]:


# Carga de librerías y lectura del archivo que contiene los datos

import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

tr = pd.read_csv('avila-tr.txt', header=None, na_values="?")
ts = pd.read_csv('avila-ts.txt', header=None, na_values="?")
tr.columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'C']
ts.columns = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'C']

data = pd.concat([tr, ts], ignore_index=True)


# # Iteración 1

# In[2]:


data


# In[3]:


data.dtypes


# In[4]:


data['C'].value_counts()


# In[5]:


data.describe()


# In[6]:


data.mode()


# In[7]:


plt.figure(figsize=(12,7))
sns.heatmap(data[['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10']].corr(),
            annot=True, linewidths=.5, annot_kws={"size": 10})


# In[8]:


data.isnull().sum()


# ## Plan para ajustar datos:
# 1. Eliminar atributo F6 en ambos conjuntos (tiene correlación alta con F10 y otros atributos)
# 2. Realizar oversampling de las clases minoritarias creando registros sintéticos con las medianas de cada clase, de tal forma que las clases minoritarias tengan por lo menos hasta la mitad de cantidad de elementos de la clase mayoritaria.
# 3. Convertir el atributo 'C' a escala numérica
# 
# El resto de los datos ya se encuentran en escala numérica, no hay atributos nulos y todos los datos se encuentran normalizados. No se realizaran más ajustes sino hasta la segunda iteración.

# In[9]:


#1. Eliminar atributo F6
data = data.drop(['F6'], axis=1)


# In[10]:


# 2. Se hace el oversampling creando un nuevo conjunto de datos 'data_bal'
major_class = data['C'].value_counts()['A']//2
data_bal = data.copy()
for C in ['F', 'E', 'I', 'X', 'H', 'G', 'D', 'Y', 'C', 'W', 'B']:
    row = data[data['C'] == C].median()
    row['C'] = C
    tmp = pd.DataFrame([row]*(major_class-data['C'].value_counts()[C]))
    data_bal = pd.concat([data_bal, tmp], ignore_index=True)
data_bal['C'].value_counts()


# In[11]:


#3. Convertir el atributo 'C' a escala numérica
u = data_bal.C.unique()
d = dict(zip(u, range(len(u))))
data_bal = data_bal.applymap(lambda s: d.get(s) if s in d else s)


# In[12]:


data_bal


# In[13]:


data_bal['C'].value_counts()


# # Entrenamiento y estimación de parámetros
# Para los scores de desempeño, se decidió escoger 'micro' en el parametro de average de la función de cada score. Esto con el fin de obtener un único resultado global entre todos los true positives, false negatives y false positives.

# In[14]:


from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from statistics import mean, stdev


# In[15]:


# holdout para MLP
HOLDOUT = 10

h_precision = []
h_recall = []
h_f1score = []

for i in range(HOLDOUT):
    X_train, X_test, y_train, y_test = train_test_split(data_bal.drop(['C'],axis=1), data_bal['C'],
                                                        test_size=0.4)
    mlp = MLPClassifier(max_iter=500)
    
    parameter_space = {
        'hidden_layer_sizes': [(50), (100)],
        'activation': ['logistic', 'tanh', 'relu'],
    }

    clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, refit=True)
    clf.fit(X_train,y_train)
    
    pred = clf.predict(X_test)
    ps = precision_score(y_test,pred, average='micro')
    rs = recall_score(y_test,pred, average='micro')
    f1s = f1_score(y_test,pred, average='micro')
    
    print("Iteración #{0}:".format(i+1))
    print()
    print(confusion_matrix(y_test, pred))
    print("Precision:",ps)
    print("Recall:",rs)
    print("F1score:",f1s)
    print("Best params", clf.best_params_)
    print()

    h_precision.append(ps)
    h_recall.append(rs)
    h_f1score.append(f1s)

print("Medias de precision, recall y f1score:",mean(h_precision),mean(h_recall),mean(h_f1score))
print("Desviaciones de precision, recall y f1score:",stdev(h_precision),stdev(h_recall),stdev(h_f1score))


# In[16]:


# holdout para SVMs
HOLDOUT = 10

h_precision = []
h_recall = []
h_f1score = []

for i in range(HOLDOUT):
    X_train, X_test, y_train, y_test = train_test_split(data_bal.drop(['C'],axis=1), data_bal['C'], test_size=0.4)
    svc = SVC(max_iter=500)
    
    parameter_space = {
        'C': [1, 10],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    }

    clf = GridSearchCV(svc, parameter_space, n_jobs=-1, cv=3, refit=True)
    clf.fit(X_train,y_train)
    
    pred = clf.predict(X_test)
    ps = precision_score(y_test,pred, average='micro')
    rs = recall_score(y_test,pred, average='micro')
    f1s = f1_score(y_test,pred, average='micro')
    
    print("Iteración #{0}:".format(i+1))
    print()
    print(confusion_matrix(y_test, pred))
    print("Precision:",ps)
    print("Recall:",rs)
    print("F1score:",f1s)
    print("Best params", clf.best_params_)
    print()

    h_precision.append(ps)
    h_recall.append(rs)
    h_f1score.append(f1s)

print("Medias de precision, recall y f1score:",mean(h_precision),mean(h_recall),mean(h_f1score))
print("Desviaciones de precision, recall y f1score:",stdev(h_precision),stdev(h_recall),stdev(h_f1score))


# ### Análisis de resultados iteración 1
# Segun el resultado de los dos holdout anteriores, se concluye que el modelo que tuvo mejor desempeño fue el de **Multi Layer Perceptron** con un score F1 en promedio del 0.96 (SVMs tuvo en promedio un F1 score del 0.84)
# 
# Un aspecto a resaltar es que en las matrices de confusión de cada iteración del holdout en el modelo MLP, la clase por la que se clasificaba más erroneamente era la clase 'A' o '0', la cual es la clase que tiene la mayor cantidad de registros no-sintéticos. Esto quiere decir que el modelo tenía cierta tendencía o bias a la hora identificar registros como pertenecientes a la clase 'A' (overfitting).

# # Iteración 2
# 
# ### Plan de ajustes para mejorar el desempeño
# Para el modelo MLP, los parametros con los que se llegó al mejor desempeño fueron la
# función de activación "tanh" y una sóla capa oculta de 100 nodos. 
# 
# 
# En el espacio de parametros tambien estuvo el parametro de una capa oculta con 50 nodos. Esto quiere sugerir que el número ideal de nodos en una capa oculta se cuentra más cercano a 100 (el número de capas aún está abierto a cambios). Las acciones a realizar para mejorar el modelo seran:
# 
# - Aproximar de forma más precisa el parametro de 'hidden_layers' con la siguiente lista de entrada: [(80), (150), (100, 100)]. Decidí escoger estos nuevos valores para verificar si aumentar o disminuir el tamaño de la cantidad de nodos en la capa oculta (100 en la primera iteración) produciría mejor desempeño. También tomé en cuenta el caso de que 100 nodos fuera la cantidad que lograra mejor desempeño, y aumenté el número de capas en uno para ver si habría mejoría y valdría la pena explorar aún más cantidad de capas ocultas.
# 
# 
# - Aproximar el parametro para optimización de pesos 'solver' del modelo, el cual no habia tenido en cuenta en la iteración anterior. Se exploraran sólo los solvers 'sgd' y 'adam', ya que según la documentación de sklearn, el solver 'lbfgs' funciona mejor con datasets pequeños.

# In[17]:


# Cubrimiento más fino con respecto al parametro de hidden_layer_sizes y exploración de parametro 'solver'

# Clasificador base
mlp = MLPClassifier(activation='tanh', max_iter=500, random_state=42)

parameter_space = {
        'hidden_layer_sizes': [(100, 100), (80), (150)],
        'solver': ['sgd', 'adam'],
    }

X_train, X_test, y_train, y_test = train_test_split(data_bal.drop(['C'],axis=1), data_bal['C'], test_size=0.4)

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, refit=True)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

print(confusion_matrix(y_test, pred))
print("precision", precision_score(y_test,pred,average='micro'))
print("recall:", recall_score(y_test,pred,average='micro'))
print("f1-score:", f1_score(y_test,pred,average='micro'))
print("best params:", clf.best_params_)


# ### Análisis de resultados iteración 2
# Se logró una mejoría con respecto a la iteración anterior de ~0.96 a ~0.99. Comparando las matrices de confusión de la iteración anterior y ésta, el modelo ahora realiza menos clasificaciones erroneas de otras clases como pertenecientes a las clase 'A' o '0'. Como se puede ver en la primer columna, los números debajo de la posición (0,0) son bastante más pequeños que en las matrices de la iteración 1 (también las clasificaciones erroneas de otras clases disminuyeron en el resto de la matriz). Esto quiere decir que el overfitting mencionado en el análisis de la iteración 1 ha disminuido.
# 
# - El parametro 'solver' no presentó cambios en cuanto a la primera iteración (ya que 'adam' es el default).
# - El número de nodos por capa que presentó mejor desempeño fue 100, y la adición de una capa extra con la misma cantidad de nodos fue el cambio que causó la mejoría en esta iteración.

# # Iteración 3
# 
# ### Plan de ajustes para mejorar el desempeño
# 
# Para esta tercera iteración, se probará realizar los siguientes ajustes:
# - Aproximar el parametro de penalización 'alpha' con los valores 0.00001, 0.0001 y 0.05.
# - Cambiar la proporción de la cantidad de datos de entrenamiento por 0.75 en lugar de 0.60
# 
# 
# Me pareció adecuado aumentar el tamaño del conjunto de entrenamiento porque en la primera iteración se realizó oversampling, lo cual aumentó el tamaño de todo el conjunto de datos de 20867 a 55718 registros (más del doble).

# In[19]:


# Clasificador base
mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='tanh', solver='adam',
                    max_iter=500, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(data_bal.drop(['C'],axis=1), data_bal['C'], test_size=0.25)

parameter_space = {
    'alpha': [0.00001, 0.0001, 0.05],
    }

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3, refit=True)
clf.fit(X_train,y_train)
pred = clf.predict(X_test)

print(confusion_matrix(y_test, pred))
print("precision", precision_score(y_test,pred,average='micro'))
print("recall:", recall_score(y_test,pred,average='micro'))
print("f1-score:", f1_score(y_test,pred,average='micro'))
print("best params:", clf.best_params_)


# ### Análisis de resultados iteración 3
# El parametro de 'alpha' no presentó cambios (el default asignado es 0.0001). Sin embargo, el aumento de tamaño del conjunto de datos logró una mejora de un F1-score de ~0.990 a ~0.995. Es decir, se obtuvo una mejora bastante mínima. Hay que resaltar que desde la iteración 2 el modelo ya presenta un nivel de precisión y recall muy elevados (de ~0.99 ambos), por lo cual poder mejorarlo más allá de unas cuantas pequeñas fracciones es una tarea compleja, sin mencionar un poco redundante, ya que el modelo mostraba un nivel de desempeño casi que perfecto.
# 
# Con respecto a la matriz de confusión, se resalta el hecho de que si bien la clase 'A' es la más erroneamente asignada, estas clasificaciones erroneas asociadas a esta clase se redujeron un poco con respecto a la iteración anterior, posiblemente por la re-partición del conjunto de datos de entrenamiento, lo cual permitió reducir el overfitting con respecto a esta clase por lo menos un poco.

# # Análisis y conclusiones generales
# Como se mencionó anteriormente, en la segunda iteración ya se demostraba un desempeño casi que perfecto en el modelo MLP. Inicialmente se había planeado realizar el balanceo de clases no en la primera iteración sino en la segunda. Sin embargo, tenía era más justo comparar ambos modelos con unos datos base que estuvieran en su mejor estado posible, ya que en el resto de iteraciones, no se modificó directamente el conjunto de datos de ninguna forma.
# 
# 
# Existen varios métodos para realizar balanceo de multi clases. Durante el proceso, se intentó usar la técnica de Synthetic Minority Over-sampling Technique for Nominal and Continuous Data (SMOTE-NC) con la libreria imblearn, pero desafortunadamente la documentación de tal librería no contenía la suficiente información detallada para usar las funciones que implementan la técnica efectivamente.
# 
# 
# Algo aprendido de este parcial es la necesidad emergente de tener un buen manejo del tiempo cuando se tienen conjuntos de datos con muchos registros, como en este caso. Solo para poder procesar ambas 10 iteraciones de ambos modelos en la primera iteración tomaba alrededor de 6 horas, y cada re-entrenamiento en las iteraciones siguientes unos 50 minutos cáda una. Esto obliga al programador a realizar pruebas y verificaciones de forma más metódica.

# In[ ]:




