from sklearn import datasets, preprocessing
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#iris tiene 
# .data(datos), 
# .target (clase de datos 0,1,2), 
# .feature_names(nombre de las columnas), 
# .target_names(nombre de las plantas)



########################## CONSTRUCCION DE DATA_FRAME IRIS ###############################

#Cargamos el conjunto de datos iris
iris = datasets.load_iris()
#Creacion de DF con la data y asignamos el nombre de las columnas con base a feature_names
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)

#Agrego la columna con la clase de planta
iris_df['class']=iris.target

#Agrego la columna con la clase de planta(nombre) en base al target asignado
iris_df['name']=iris.target_names[iris.target]

########################## FIN CONSTRUCCION DE DATA_FRAME IRIS ###############################



########################## LIMPIEZA DE DATA FRAME ###############################

#Eliminamos datos que son NA
iris_df = iris_df.dropna()
#Reestablecemos el indice y eliminamos el indice anterior (drop true)
iris_df = iris_df.reset_index(drop=True)

# Guardo y elimino datos por aparte
names = iris_df['name']
target = iris_df['class']

iris_df = iris_df.drop(['name', 'class'], axis=1) #1==columnas, 0==filas

########################## FIN LIMPIEZA DE DATA FRAME ###############################


########################## NORMALIZACION DE DATOS ###############################

min_max_scaler = preprocessing.MinMaxScaler()
#Ajustamos y transformamos los datos
df_escalado = min_max_scaler.fit_transform(iris_df) #El resultado es un array de NumPy
df_escalado = pd.DataFrame(df_escalado) #Convertimos el array a un data frame
df_escalado = df_escalado.rename(columns= {0:'sepal length', 1:'sepal width', 2:'petal length', 3:'petal width'} )
# print(iris_df.head())
# print(df_escalado.head())

########################## FIN NORMALIZACION DE DATOS ###############################


########################## METODO DEL CODO ###############################
# Sirve para determinar el número óptimo de clusters en un algoritmo de K-Means

wcss = [] # Es una lista vacía que almacenará los valores de Within-Cluster-Sum of Squares (WCSS), 
          # también conocido como inercia. La inercia mide la suma de las distancias al cuadrado de 
          # cada punto a su centroide más cercano. Un valor bajo de WCSS indica que los puntos están 
          # más cerca de sus centroides, lo que sugiere un mejor agrupamiento.

#For para probar diferentes cantidades de clusters
for i in range(2,11):
    # kmeans = KMeans(n_clusters=i, init='random', max_iter=300, n_init=10, random_state=0) 
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=1, random_state=0) 
    #init selecciona el metodo a usar para seleccionar los centroides iniciales, max_iter limita las iteraciones por cada inicializacion
    # n_init nos indica el total de inicializaciones que tendremos, random_state fija la semilla de aleatoriedad
    # al usar k-means++ no se recomeinda usar random_state con un n_init > 1
    # en caso contrario probablemente cada iteracion dara el mismo resultado causando redundancia 

    #Ajustamos el modelo a los datos (entrenar el modelo)
    kmeans.fit(df_escalado)

    #Calcular y almacenar la inercia (WCSS)
    wcss.append(kmeans.inertia_)

    print(f'Numero de clusters: {i}')
    print(f'Inercia calculada: {kmeans.inertia_}')

plt.plot(range(2,11), wcss, marker='o', color='red')
plt.title('Metodo de Elbow')
plt.xlabel('Numero de Clusters')
plt.ylabel('WCSS')
plt.show()

########################## FIN METODO DEL CODO ###############################


########################## ENTRENAR MODELO CON NUMERO DE CLUSTERS SELECCIONADO ###############################

#Aqui inicializo mas veces sin importar la redundancia ya que busco que se ajuste lo mas posible
kmeans = KMeans(n_clusters=int(input('Ingresa el numero de Clusters: ')), init='k-means++', n_init=10, max_iter=300, random_state=0 )
kmeans.fit(df_escalado)
#print(f'--------------------------\nInercia calculada: {kmeans.inertia_}')

########################## FIN ENTRENAR MODELO CON NUMERO DE CLUSTERS SELECCIONADO ###############################



########################## GRAFICAS ###############################

#Obtenemos los label de cada cluster
cluster_labels = kmeans.labels_
#print(f'Cluster labels: {cluster_labels}')

#Definimos los colores para cada cluster
diccionario_colores = {0:'black', 1:'purple', 2:'blue', 3:'orange', 4:'pink', 5:'yellow'}
colores_clusters = [diccionario_colores[i] for i in cluster_labels]
#print(colores_clusters)

#Visualizacion de colores y especies
indice = 0
for i in iris.target_names:
    print(f'{i} : {diccionario_colores.get(indice)}')
    indice += 1

#Graficamos los clusters
plt.scatter(df_escalado.iloc[:, 0], df_escalado.iloc[:, 1], c=colores_clusters, s=50, alpha=0.5)
#Graficamos los centroides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroides')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title(f'K-means Clustering')
plt.show()

plt.scatter(iris_df.iloc[:, 0], iris_df.iloc[:, 1], c=colores_clusters, s=50, alpha=0.5)
plt.show()
########################## FIN GRAFICAS ###############################



# print(names, target)

# print(len(iris_df))
# print(f'{iris_df.head()}')

#print(f'Los nombres de las plantas son: {plantas}')