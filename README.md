# Documentación del Código: Análisis de Clustering con K-Means en el Conjunto de Datos Iris

## Introducción
Este código realiza un análisis de clustering en el conjunto de datos Iris utilizando el algoritmo de K-Means. A continuación, se documenta el flujo del código y la razón de cada sección.

---

## 1. Importación de Librerías
```python
from sklearn import datasets, preprocessing
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
```
**Propósito:** Importar las librerías necesarias para la manipulación de datos (pandas), el aprendizaje automático (scikit-learn) y la visualización (matplotlib).

---

## 2. Carga del Conjunto de Datos Iris
```python
iris = datasets.load_iris()
```
**Propósito:** Cargar el conjunto de datos Iris, que contiene 150 muestras de tres especies de flores (Setosa, Versicolor, Virginica), cada una con 4 características.

---

## 3. Creación del DataFrame
```python
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['class'] = iris.target
iris_df['name'] = iris.target_names[iris.target]
```
**Propósito:**
- Convertir los datos en un DataFrame de pandas.
- Agregar columnas para representar la clase numérica (`target`) y el nombre de la especie (`name`).

---

## 4. Limpieza del DataFrame
```python
iris_df = iris_df.dropna()
iris_df = iris_df.reset_index(drop=True)
names = iris_df['name']
target = iris_df['class']
iris_df = iris_df.drop(['name', 'class'], axis=1)
```
**Propósito:**
- Eliminar valores nulos (aunque el dataset Iris no los contiene por defecto).
- Restablecer los índices tras la eliminación de datos.
- Guardar las etiquetas de las especies en variables separadas.
- Eliminar columnas que no se usarán en el clustering.

---

## 5. Normalización de los Datos
```python
min_max_scaler = preprocessing.MinMaxScaler()
df_escalado = min_max_scaler.fit_transform(iris_df)
df_escalado = pd.DataFrame(df_escalado, columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
```
**Propósito:**
- Aplicar escalado Min-Max para normalizar las características entre 0 y 1, mejorando el rendimiento de K-Means.

---

## 6. Método del Codo para Selección de Clusters
```python
wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=1, random_state=0)
    kmeans.fit(df_escalado)
    wcss.append(kmeans.inertia_)
    print(f'Numero de clusters: {i}')
    print(f'Inercia calculada: {kmeans.inertia_}')
```
**Propósito:**
- Determinar el número óptimo de clusters evaluando la inercia (suma de distancias al cuadrado entre los puntos y sus centroides).
- Se grafica la inercia frente al número de clusters para visualizar el "codo" en la curva.

```python
plt.plot(range(2, 11), wcss, marker='o', color='red')
plt.title('Metodo de Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WCSS')
plt.show()
```

---

## 7. Entrenamiento del Modelo K-Means
```python
kmeans = KMeans(n_clusters=int(input('Ingresa el numero de Clusters: ')), init='k-means++', n_init=10, max_iter=300, random_state=0)
kmeans.fit(df_escalado)
```
**Propósito:**
- Entrenar el modelo de K-Means con el número de clusters determinado por el usuario.
- Se ajusta a los datos normalizados para formar agrupaciones.

---

## 8. Visualización de los Clusters
```python
cluster_labels = kmeans.labels_
diccionario_colores = {0:'black', 1:'purple', 2:'blue', 3:'orange', 4:'pink', 5:'yellow'}
colores_clusters = [diccionario_colores[i] for i in cluster_labels]
```
**Propósito:**
- Obtener etiquetas de cluster asignadas a cada punto.
- Asignar colores a cada cluster.

```python
plt.scatter(df_escalado.iloc[:, 0], df_escalado.iloc[:, 1], c=colores_clusters, marker='o', edgecolor='k')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Clusters del conjunto de datos Iris')
plt.show()
