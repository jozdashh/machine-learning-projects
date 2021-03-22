#!/usr/bin/env python
# coding: utf-8

# In[71]:


import numpy as np
import pandas as pd
import sklearn as sk
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data = pd.read_csv('household_power_consumption.txt', sep=";", header=None, na_values="?", skiprows=1)

data.columns = ['Date','Time','Global_active_power','Global_reactive_power','Voltage','Global_intensity',
                'Sub_metering_1','Sub_metering_2','Sub_metering_3']
data


# In[72]:


# Tipo de los atributos
data.dtypes


# In[73]:


# Medidas de centralidad para atributos numéricos
data.describe()


# In[74]:


data.mode()


# In[75]:


# Correlación entre los datos
sns.heatmap(data.corr(), square=True, annot=True)


# In[76]:


# Existen datos atípicos?

fig1, bp1 = plt.subplots()
bp1.set_title('Global_active_power')
bp1.boxplot(data['Global_active_power'].dropna())

fig1, bp1 = plt.subplots()
bp1.set_title('Global_reactive_power')
bp1.boxplot(data['Global_reactive_power'].dropna())

fig1, bp1 = plt.subplots()
bp1.set_title('Voltage')
bp1.boxplot(data['Voltage'].dropna())

fig1, bp1 = plt.subplots()
bp1.set_title('Global_intensity')
bp1.boxplot(data['Global_intensity'].dropna())

fig1, bp1 = plt.subplots()
bp1.set_title('Sub_metering_1')
bp1.boxplot(data['Sub_metering_1'].dropna())

fig1, bp1 = plt.subplots()
bp1.set_title('Sub_metering_2')
bp1.boxplot(data['Sub_metering_2'].dropna())

fig1, bp1 = plt.subplots()
bp1.set_title('Sub_metering_3')
bp1.boxplot(data['Sub_metering_3'].dropna())


# In[77]:


# Registros faltantes por cada atributo
data.isnull().sum()


# In[78]:


# Cantidad máxima de atributos faltantes en un mismo registro
max(data.isnull().sum(axis=1))


# In[79]:


# Cantidad de registros que tienen 7 atributos faltantes
data.isna().sum(axis=1)[data.isna().sum(axis=1) == 7].size


# ### Plan para ajustar los datos
# 1. Eliminar los 25979 registros que tienen los 7 atributos faltantes.
# 2. Concatenar las columnas 'Date' y 'Time' en 'DateTime'
# 3. Transformar los atributo 'DateTime' al dtype datetime.
# 4. Eliminar los atributos 'Date' y 'Time'.
# 5. Extraer la información numérica de cada fecha y hora (año, mes, día de la semana, hora, etc) y asignar cada una a una columna nueva para cada registro.
# 6. Eliminar columna 'DateTime'.

# In[80]:


# 1. Eliminar los 25979 registros que tienen los 7 atributos faltantes.
data = data.dropna()


# In[81]:


# 2. Concatenar las columnas 'Date' y 'Time' en 'DateTime'
data['DateTime'] = data[['Date', 'Time']].agg('-'.join, axis=1)


# In[82]:


# 3. Transformar los atributo 'DateTime' al dtype datetime.
data['DateTime'] = pd.to_datetime(data['DateTime'], format='%d/%m/%Y-%H:%M:%S')


# In[83]:


data


# In[84]:


# 4. Eliminar los atributos 'Date' y 'Time'.
data = data.drop(['Date'], axis=1)
data = data.drop(['Time'], axis=1)


# In[85]:


# 5. Extraer la información numérica de cada fecha y hora
data['Year'] = data['DateTime'].dt.year
data['Month'] = data['DateTime'].dt.month
data['Week'] = data['DateTime'].dt.week
data['DayofWeek'] = data['DateTime'].dt.dayofweek
data['Hour'] = data['DateTime'].dt.hour


# In[86]:


# 6. Eliminar columna 'DateTime'.
data = data.drop(['DateTime'], axis=1)


# In[87]:


data = data.reset_index(drop=True)


# In[88]:


data


# In[120]:


# Se reduce el tamaño del conjunto de datos. De lo contrario las metricas de desempeño toman muchas horas-
# -cada una en poder ser calculadas. Sin mencionar que con más de 20'000 registros, el método jerarquico
# se desborda en reserva de memoria en la maquina en la que estoy trabajando (trata de alocar 60+GB de RAM)
# y en el DBSCAN, se genera un sólo conglomerado gigante de los datos (también con 20'000 registros o más).
train, test = train_test_split(data, train_size=0.001, random_state=42)
train.shape


# # 1. Clustering por K-means

# In[121]:


from sklearn.cluster import KMeans

# Se realizan 15 conglomerados y se grafica el error cuadrático
sse = {}
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=42, n_jobs=-1).fit(train)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[122]:


# Se escoge K = 7 para el número de conglomerados
# Se muestran las coordenadas de cada centroide de los 7 grupos
kmeans = KMeans(n_clusters=7, random_state=42, n_jobs=-1).fit(train)
cc = pd.DataFrame(kmeans.cluster_centers_)
cc.columns = ['Global_active_power','Global_reactive_power','Voltage','Global_intensity',
                'Sub_metering_1','Sub_metering_2','Sub_metering_3', 'Year', 'Month', 
              'Week', 'DayofWeek', 'Hour']
cc


# # 2. Clustering por Método jerárquico

# In[123]:


from sklearn.cluster import AgglomerativeClustering
agglo = AgglomerativeClustering(n_clusters=7, linkage='average').fit(train)


# # 3. Clustering por DBSCAN

# In[124]:


from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

nbrs = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(train)


# In[125]:


# Se realiza el gráfico para determinar los parametros del radio de la circunferencia (eps) y el umbral
distances, indices = nbrs.kneighbors(train)
distances = [np.linalg.norm(e) for e in distances]
distances.sort()

plt.figure()
plt.plot(distances, 'b.', markersize=0.3)
plt.xlabel("Datos")
plt.ylabel("Umbral")
plt.show()


# In[126]:


distances[1500]


# In[128]:


# Se escoge entonces eps (radio de la circunferencia) como distances[1500] y min_samples (umbral) como 7
dbscan = DBSCAN(eps=distances[1500], min_samples=7).fit(train)


# ## Comparación y análisis de las técnicas

# In[129]:


aux_train = train.copy()


# In[132]:


# Calcular la distancia entre dos vectores de coordenadas o registros
def proximidad(x, y):
    return np.linalg.norm(x-y)

# Calcular la cohesion entre todos los elementos de un conglomerado
def cohesion(Ci_list, dt):
    ans = 0
    for i in Ci_list:
        x = dt.iloc[i].to_numpy()
        for j in Ci_list:
            ans += proximidad(x, dt.iloc[j].to_numpy())
    return ans

# Calcular la separacion entre dos conglomerados
def separacion(Ci_list, Cj_list, dt):
    ans = 0
    for i in Ci_list:
        x = dt.iloc[i].to_numpy()
        for j in Cj_list:
            ans += proximidad(x, dt.iloc[j].to_numpy())
    return ans


# In[133]:


# Calculamos cohesion y separacion para K-means
aux_train['labels'] = kmeans.labels_
aux = aux_train.groupby('labels')
ans1, ans2 = 0, 0

for k in range(7):
    Ci_list = aux.groups[k].to_numpy()
    ans1 += cohesion(Ci_list, data)
print("Cohesion:", ans1)
kmeans_coh = ans1

for k in range(7):
    Ci_list = aux.groups[k].to_numpy()
    for l in range(7):
        if k != l:
            Cj_list = aux.groups[l].to_numpy()
            ans2 += separacion(Ci_list, Cj_list, data)
print("Separacion:", ans2)
kmeans_sep = ans2


# In[134]:


# Calculamos cohesion y separacion para Método jerárquico
aux_train['labels'] = agglo.labels_
aux = aux_train.groupby('labels')
ans1, ans2 = 0, 0

for k in range(7):
    Ci_list = aux.groups[k].to_numpy()
    ans1 += cohesion(Ci_list, data)
print("Cohesion:", ans1)
jerarq_coh = ans1

for k in range(7):
    Ci_list = aux.groups[k].to_numpy()
    for l in range(7):
        if k!= l:
            Cj_list = aux.groups[l].to_numpy()
            ans2 += separacion(Ci_list, Cj_list, data)
print("Separacion:", ans2)
jerarq_sep = ans2


# In[ ]:


# Cantidad de conglomerados generada por DBSCAN (6, con indices 0..5)
max(dbscan.labels_)


# In[141]:


# Calculamos cohesion y separacion para DBSCAN
aux_train['labels'] = dbscan.labels_
aux = aux_train.groupby('labels')
ans1, ans2 = 0, 0

for k in range(6):
    Ci_list = aux.groups[k].to_numpy()
    ans1 += cohesion(Ci_list, data)
print("Cohesion:", ans1)
dbscan_coh = ans1

for k in range(6):
    Ci_list = aux.groups[k].to_numpy()
    for l in range(6):
        if k!= l:
            Cj_list = aux.groups[l].to_numpy()
            ans2 += separacion(Ci_list, Cj_list, data)
print("Separacion:", ans2)
dbscan_sep = ans2


# In[146]:


# Qué método tuvo la mejor cohesión en cada conglomerado?
min(kmeans_coh, jerarq_coh, dbscan_coh)


# In[147]:


# Qué método tuvo el mayor separación entre conglomerados?
max(kmeans_sep, jerarq_sep, dbscan_sep)


# El modelo de K-means, por ser más eficiente computacionalmente, pudo haberse entrenado con el conjunto de datos completo (a diferencia del jerarquico que se desbordaba por memoria, y de que el DBSCAN sólo creara un sólo conglomerado). Pero, el hacer esto no hace justa la comparacion entre los tres métodos.
# 
# El método que tuvo mejor **cohesión** fue K-means, y el que tuvo mejor **separación**, también, fue el método de K-means. Obviamente se destaca el hecho de que no se está trabajando con el conjunto entero de datos, las métricas usadas no son del todo precisas y de que posiblemente otras métricas de desempeño pudieron ser usadas para comparar los tres métodos de tal forma que se pudiera apreciar mejor las fortalezas y debilidades de cáda uno.
# 
# Sin embargo, bajo las condiciones de poder cómputo en las que se desarrolló el taller y las métricas usadas, el método de conglomerados que mejor desempeño tuvo fue el de **K-means**, ya que este pudo formar grupos cuyos puntos estuvieran más cerca los unos a los otros (lo cual sugiere que pudo identificar los patrones de cada grupo en el conjunto de datos de forma correcta), y a su vez, estos grupos se encontraban más alejados o aislados entre sí que en cualquiera de los otros métodos.

# In[ ]:




