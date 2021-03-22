#!/usr/bin/env python
# coding: utf-8

# # Parcial 1 Aprendizaje Automático y Análisis de Datos
# - Autor: Josue Peña Atencio - 8935601
# - Fecha: Febrero 28 2020

# In[1]:


import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_rows = 100
pd.options.display.max_columns = 35

from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

url="https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/dis.data"
data = pd.read_csv(url, header=None, true_values='t', false_values='f', na_values='?')

data.columns = ['Age', 'Sex', 'On_thyroxine', 'Q_on_thyroxine', 'On_antithyroid_med', 'Sick',
                'Pregnant', 'Thyroid_surgery', 'I131_treatment', 'Q_hypothyroid',
                'Q_hyperthyroid', 'Lithium', 'Goitre', 'Tumor', 'Hypopituitary', 'Psych', 'TSH_measured',
                'TSH', 'T3_measured', 'T3', 'TT4_measured', 'TT4', 'T4U_measured', 'T4U', 'FTI_measured',
                'FTI', 'TBG_measured', 'TBG', 'Source', 'V'] # V for veredict

# Al final de cada línea, hay información extra sobre el id de cada paciente.
# Esto imposibilita que hayan sólo 2 atributos de salida, ademas de no ser relevante para este contexto.
data['V'] = data['V'].apply(lambda x: x.split('.')[0])

data


# # 1. Conocer el conjunto de datos y realizar un plan para ajustarlos

# In[2]:


# Número de registros y atributos
data.shape


# In[3]:


# Tipo de los atributos
data.dtypes


# In[4]:


# Medidas de centralidad para atributos numéricos
data.describe()


# In[5]:


# Medidas de centralidad para atributos categóricos y numéricos
data.mode()


# In[6]:


# Existen datos atípicos?

fig1, bp1 = plt.subplots()
bp1.set_title('Age')
bp1.boxplot(data['Age'].dropna())

fig1, bp2 = plt.subplots()
bp2.set_title('TSH')
bp2.boxplot(data['TSH'].dropna())

fig1, bp3 = plt.subplots()
bp3.set_title('T3')
bp3.boxplot(data['T3'].dropna())

fig1, bp4 = plt.subplots()
bp4.set_title('TT4')
bp4.boxplot(data['TT4'].dropna())

fig1, bp5 = plt.subplots()
bp5.set_title('T4U')
bp5.boxplot(data['T4U'].dropna())

fig6, bp6 = plt.subplots()
bp6.set_title('FTI')
bp6.boxplot(data['FTI'].dropna())


# In[7]:


# Registros con 'Age' > 400
data[data['Age'] > 400]


# In[8]:


# Registros con 'TSH' > 400
data[data['TSH'] > 400]


# In[9]:


# Registros con 'T3 > 10
data[data['T3'] > 10]


# In[10]:


# Registros con 'FTI' > 300
data[data['FTI'] > 300]


# In[11]:


# Correlación entre los datos
sns.heatmap(data[['Age','TSH','T3','TT4','T4U','FTI','TBG']].corr(), square=True, annot=True)


# In[12]:


# Registros faltantes por cada atributo
data.isnull().sum()


# In[13]:


# Cantidad máxima de atributos faltantes en un mismo registro
max(data.isnull().sum(axis=1))


# In[14]:


# Cantidad de registros que tienen 5 o más atributos faltantes
data.isna().sum(axis=1)[data.isna().sum(axis=1) > 4].size


# In[15]:


# Cantidad máxima de valores nulos por registro en clase 'discordant'
max(data[data['V']=='discordant'].isna().sum(axis=1))


# In[16]:


# Cantidad de registros con atributos binarios == True
for col in data:
    if data[col].dtype == 'bool':
        print(col,data[data[col] == True].shape[0])


# In[17]:


# Registros con 'Lithium' == True
data[data['Lithium'] == True]


# In[18]:


# Registros con 'Goitre' == True
data[data['Goitre'] == True]


# In[19]:


# Registro con 'Hypopituitary' == True
data[data['Hypopituitary'] == True]


# In[20]:


# Cantidad de registros por cada clase o atributo de salida
data['V'].value_counts()


# ### Plan para ajustar los datos
# 1. **Eliminar todos los atributos '*_measured'**. Estos sólo indican si su valor medido asociado es nulo o no. Son redundantes (ademas de poner ser inconsistentes, eg que en un registro 'TSH_measured' sea False pero 'TSH' sea no-nulo, y viceversa)
# 
# 
# 2. **Eliminar el el atributo 'TBG'**. Todos los valores de esa columna son nulos.
# 
# 
# 3. **Eliminar el atributo 'source'**. Este sólo indica la fuente referida de los datos.
# 
# 
# 4. **Eliminar el atributo 'TT4'**. Este tiene niveles muy altos de correlación con los atributos 'FTI' (0.8), 'T4U' (0.43) y 'T3' (0.56). Se escoge este atributo en lugar de 'FTI' ya que éste ultimo sólo tiene correlación alta con 'TT4' y 'T3'.
# 
# 
# 5. **Eliminar los atributos 'Lithium', 'Goitre' y 'Hypopituitary'**. Son atributos binarios, y la presencia de valores 'True' es muy poca para ser significativa (No más del 2 o 3% de registros cuentan con valores True en alguno de estos atributos). Además y no menos importante, ninguno de los registros pertenece a la clase 'discordant'.
# 
# 
# 6. **Borrar los registros que tengan más de 2 atributos nulos**. Como los registros de la clase 'discordant' solo tienen a lo sumo 2 atributos nulos (después de hacer la eliminación de atributos), ningún registro que pertenezca a esa clase es borrado.
# 
# 
# 7. **Reemplazar los atributos booleanos nulos por sus modas**.
# 8. **Reemplazar los atributos numéricos por sus medianas**.
# 
# __Para los pasos 9-12__: Todos los registros atípicos que serán reemplazados no pertenecen a la clase 'discordant', entonces no hay peligro de perder información limitada e importante para esa clase.
# 
# 9. **Reemplazar el dato atípico de 'Age' > 400 con la moda**.
# 10. **Reemplazar los datos atípicos de 'TSH' > 400 con el primer valor < 400**.
# 11. **Reemplazar el dato atípico de 'T3' > 10 con el primer valor < 10**.
# 12. **Reemplazar los datos atípicos de 'FTI' > 300 con el primer valor < 300**.
# 
# 
# 13. **Normalizar los atributos TSH, T3, T4U y FTI**.
# 14. **Convertir los atributos categóricos a escala numérica**.
# 
# 
# 15. **Hacer balanceo de clases 1:1 con oversampling de la clase minoritaria**, **DESPUÉS** de separar el conjunto de datos en conjunto de entrenamiento y de pruebas. Esto es para prevenir que ocurra overfitting, ya que si se hace antes habrán observaciones exactamente iguales en ambos conjuntos, causando poca generalidad en los datos y métricas de desempeño casi que perfectas.

# # 2. Preprocesamiento del conjunto de datos

# In[21]:


#1. Eliminar todos los atributos '*_measured'.
data.drop(['TSH_measured'], axis=1, inplace=True)
data.drop(['T3_measured'], axis=1, inplace=True)
data.drop(['TT4_measured'], axis=1, inplace=True)
data.drop(['T4U_measured'], axis=1, inplace=True)
data.drop(['FTI_measured'], axis=1, inplace=True)
data.drop(['TBG_measured'], axis=1, inplace=True)

#2. Eliminar el atributo 'TBG'.
data.drop(['TBG'], axis=1, inplace=True)

#3. Eliminar el atributo 'source'.
data.drop(['Source'], axis=1, inplace=True)

#4. Eliminar el atributo 'TT4'
data.drop(['TT4'], axis=1, inplace=True)

#5 Eliminar los atributos 'Lithium', 'Goitre' y 'Hypopituitary'
data.drop(['Lithium'], axis=1, inplace=True)
data.drop(['Goitre'], axis=1, inplace=True)
data.drop(['Hypopituitary'], axis=1, inplace=True)


# In[22]:


# Cantidad máxima de valores nulos por registro en clase 'discordant' luego de eliminar atributos
max(data[data['V']=='discordant'].isna().sum(axis=1))


# In[23]:


# Cantidad de registros con más de 2 atributos nulos
data.isna().sum(axis=1)[data.isna().sum(axis=1) > 2].size


# In[24]:


#6. Borrar los registros que tengan más de 2 atributos nulos
data.dropna(axis = 0, thresh = 16, inplace=True)

#7. Reemplazar los atributos nulos por sus modas.
am = data.mode()['Age'][0]
sm = data.mode()['Sex'][0]
values1 = {'Age': am, 'Sex': sm}
data.fillna(value=values1, inplace=True)

#8. Reemplazar los atributos numéricos por sus medianas
tshm = data.median()['TSH']
t3m = data.median()['T3']
t4um = data.median()['T4U']
ftim = data.median()['FTI']
values2 = {'TSH': tshm, 'T3': t3m, 'T4U': t4um, 'FTI': ftim}
data.fillna(value=values2, inplace=True)


# In[25]:


#9. Reemplazar el dato atípico de 'Age' > 400 con la moda.
data.loc[data['Age']>400,'Age'] = am

#10. Reemplazar los datos atípicos de 'TSH' > 400 con el primer valor < 400.
data.loc[data['TSH']>400,'TSH'] = data[data['TSH']<400].sort_values(by=['TSH'], ascending=False).iloc[0]['TSH']

#11. Reemplazar el dato atípico de 'T3' > 10 con el primer valor < 10.
data.loc[data['T3']>10,'T3'] = data[data['T3']<10].sort_values(by=['T3'], ascending=False).iloc[0]['T3']

#12. Reemplazar los datos atípicos de 'FTI' > 300 con el primer valor < 300.
data.loc[data['FTI']>300,'FTI'] = data[data['FTI']<300].sort_values(by=['FTI'], ascending=False).iloc[0]['FTI']


# In[26]:


#13. Convertir los atributos categóricos a escala numérica

uV = data.V.unique()
VDict = dict(zip(uV, range(len(uV))))
data = data.applymap(lambda s: VDict.get(s) if s in VDict else s)

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)


# In[27]:


#14. Normalizar los atributos TSH, T3, T4U y FTI.
data['TSH'] = preprocessing.robust_scale(data['TSH'])
data['T3'] = preprocessing.robust_scale(data['T3'])
data['T4U'] = preprocessing.robust_scale(data['T4U'])
data['FTI'] = preprocessing.robust_scale(data['FTI'])


# In[28]:


data


# In[29]:


# Dimensiones del conjunto de datos luego de hacer preprocesamiento
data.shape


# In[30]:


data['V'].value_counts()


# # 3. Separar el conjunto de datos en conjunto de entrenamiento y de prueba

# In[31]:


# Separación con base en el conjunto de datos alterno con oversampling
IN_train, IN_test = train_test_split(data, test_size = 0.25, random_state = 123, shuffle=True)


# In[32]:


g1 = IN_train.groupby('V')
g2 = IN_test.groupby('V')


# In[33]:


#15. Hacer balanceo de clases 1:1 con oversampling de la clase minoritaria.

## Balanceo conjunto de datos de entrenamiento
# Separar clases de datos
major_class = IN_train[data.V==0].copy()
minor_class = IN_train[data.V==1].copy()

# Relizar oversampling
minor_class_upsampled = resample(minor_class, replace=True, n_samples=g1.size().max(), random_state=435)

# Mezclar clase mayoritaria 
IN_train = pd.concat([major_class, minor_class_upsampled])

## Balanceo conjunto de datos de prueba
major_class = IN_test[data.V==0].copy()
minor_class = IN_test[data.V==1].copy()
minor_class_upsampled = resample(minor_class, replace=True, n_samples=g2.size().max(), random_state=435)
IN_test = pd.concat([major_class, minor_class_upsampled])


# In[34]:


IN_train['V'].value_counts()


# In[35]:


IN_test['V'].value_counts()


# In[36]:


OUT_train, OUT_test = IN_train['V'].copy(), IN_test['V'].copy()
IN_train.drop(['V'], axis=1, inplace=True)
IN_test.drop(['V'], axis=1, inplace=True)


# # a. Clasificador por regresión lineal
# Para este clasificador, no tiene sentido calcular la matriz de confusión y las métricas de desempeño, ya que los valores predecidos son regresiones (números continuos). Ajustar las predicciones para que caigan en alguna de las dos clases implica agregar una función de activación, lo cual es lo mismo que aplicar el clasificador por regresión logística.
# 
# 
# Se podría realizar algún tipo de redondeo para forzar a que cada predicción caiga en alguna de las dos clases, sin embargo tal decisión es más bien poco precisa y los resultados por lo tanto no lo serían tampoco.

# In[37]:


reg = LinearRegression().fit(IN_train, OUT_train)
pred1 = reg.predict(IN_test)


# In[38]:


pred1


# # b. Clasificador por regresión logística

# In[39]:


clf = LogisticRegression(max_iter=1000).fit(IN_train, OUT_train)
pred2 = clf.predict(IN_test)


# In[40]:


# Matriz de confusión
confusion_matrix(OUT_test, pred2)


# In[41]:


# Precision
precision_score(OUT_test, pred2)


# In[42]:


# Recall
recall_score(OUT_test, pred2)


# In[43]:


# F1-score
f1_score(OUT_test, pred2)


# # c. K-vecinos más cercanos (k=5)

# In[44]:


neigh = KNeighborsClassifier(n_neighbors=5).fit(IN_train, OUT_train)
pred3 = neigh.predict(IN_test)


# In[45]:


# Matriz de confusión
confusion_matrix(OUT_test, pred3)


# In[46]:


# Precision
precision_score(OUT_test, pred3)


# In[47]:


# Recall
recall_score(OUT_test, pred3)


# In[48]:


# F1-score
f1_score(OUT_test, pred3)


# # d. Análisis discriminante lineal

# In[49]:


lda = LinearDiscriminantAnalysis().fit(IN_train, OUT_train)
pred4 = neigh.predict(IN_test)


# In[50]:


# Matriz de confusión
confusion_matrix(OUT_test, pred4)


# In[51]:


# Precision
precision_score(OUT_test, pred4)


# In[52]:


# Recall
recall_score(OUT_test, pred4)


# In[53]:


# F1-score
f1_score(OUT_test, pred4)


# # e. Análisis discriminante cuadrático

# In[54]:


qda = QuadraticDiscriminantAnalysis().fit(IN_train, OUT_train)
pred5 = neigh.predict(IN_test)


# In[55]:


# Matriz de confusión
confusion_matrix(OUT_test, pred5)


# In[56]:


# Precision
precision_score(OUT_test, pred5)


# In[57]:


# Recall
recall_score(OUT_test, pred5)


# In[58]:


# F1-score
f1_score(OUT_test, pred5)


# # Análisis de resultados

# La motivación principal en el diseño del plan de acción fue reducir la dimensionalidad del conjunto de datos, al éste tener tan pocos registros (solo 2800). Esto con el fin de poder sacarle el mejor provecho a esta cantidad reducida de datos. Sin embargo, tal decisión influye en los resultados arrojados por los modelos, especialmente en los más sofisticados.

# 0: Clase de datos 'negative' o 'negativa'
# 
# 
# 1: Clase de datos 'discordant' o 'positiva'
# 
# ### Fortalezas y debilidades con respecto al conjunto de datos
# 
# ###### Clasificador por regresión logística
# - El modelo muestra buen desempeño en sus predicciones para este conjunto de datos: Según la matriz de confusión, clasificó correctamente 580 instancias pertenecientes a la clase positiva, y sólo 60 incorrectamente. Para el caso de la clase negativa, clasificó 552 instancias correctamente y 88 incorrectamente. El modelo es más débil con respecto a la clasificación de las clases negativas como tal, pero más fuerte al clasificar instancias de la clase positiva.
# 
# 
# Se elaborará más sobre el desempeño de este modelo en las comparaciones con los otros.
# 
# 
# 
# ###### K-vecinos más cercanos (k=5), Análisis discriminante lineal, Análisis discriminante cuadrático
# Estos tres modelos, extrañamente, generaron todos una matriz de confusión exactamente igual (87 true positives, 617 true nagatives, etc). Como se puede observar, los modelos son más fuertes a la hora de clasificar una instancia en la clase negativa de forma correcta, pero son muy débiles para correctamente clasificar instancias positivas (sólo 87 true positives y 553 false negatives). Es decir, los modelos generan demasiados falsos positivos, lo cual en este contexto de veredictos frente a diagnosticos para enfermedades de la tiroides, pueden ser muy peligrosos.
# 
# 
# El factor principal que pudo haber afectado el desempeño de estos modelos un poco más robustos que el regresor lineal logístico es la estrategia utilizada para hacer el balanceo entre clases. Se utilizó oversampling, es decir generar más atributos de la clase minoritaria de forma sintética. Además, en el caso de los atributos de tipo flotante sobre las mediciones hormonales, existen relaciones lineales (y esto se ve reflejado en el heatmap de correlación), lo cual puede ser un problema en el análisis discriminante cuadrático.
# 
# 
# Como último factor a considerar es el hecho de la eliminación de los tres atributos binarios Lithium, Goitre y Hypopituitary. Si bien estos sólo eran positivos en registros de la clase mayoritaria (lo cual sugiere que con tal información extra, se podría aumentar el número de true negatives), puede que los modelos hubieran tenido un mejor desempeño dejando tales atributos, con la condicion de tener un conjunto de datos más grande, al cual no se le tenga que hacer tanto oversampling.

# ### Comparación entre las técnicas
# Este conjunto de datos tiene un número de registros limitado, mientras que la cantidad de atributos es muy numerosa (30 sin hacer preprocesamiento). Aún después de borrar 12 atributos, los clasificadores más sofisticados (analisis lineal, cuadrático y vecinos más cercanos) tienen un score de recall o sensitivity muy bajo (0.135), es decir que la probabilidad de que los modelos identifiquen el perfil de un paciente como veredicto 'discordant' entre aquellas personas que tengan tal veredicto realmente es muy poca.
# 
# 
# Esto puede deberse al hecho de que entre mayor es la cantidad de dimensiones en un conjunto de datos, se requieren muchos más datos de entrenamiento para minimizar los errores de clasificación.
# 
# 
# Por otro lado, estos últimos tres modelos demostraron un desempeño aceptable en cuanto a su capacidad para clasificar correctamente una instancia realmente verdadera como como tal entre todas las clasificadas como verdaderas (precision 0.79). Aunque, esta proporción no es sorprendente sabiendo sobre la poca cantidad de true positives que los modelos generaron.
# 
# El F1 score da suficiente información para poder evaluar el desempeño general bajo las dos métricas de precision y recall. El F1 score en los modelos es de sólo ~0.232. En conclusión, para este conjunto de datos, los tres modelos tienen un muy mal desempeño.
# 
# 
# A comparación, el clasificador por regresión logistica demuestra un muy buen desempeño a la hora de clasificar correctamente instancias pertenecientes a la clase positiva de 'discordant' entre todas las clasificadas como pertenecientes a tal clase (precision de ~0.868), ademas de tener aún mejor desempeño con respecto a la probabilidad de clasificar una instancia positiva correctamente entre todas las instancias que pertenecen realmente a tal clase (recall de ~0.906).
# 
# ###### Conclusión: 
# El modelo de clasificación por regresión lineal logística tiene el mejor desempeño entre todos los modelos, para este conjunto de datos.
