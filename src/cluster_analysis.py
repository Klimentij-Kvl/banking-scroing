import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def elbow_method(data: pd.DataFrame):
    # Метод локтя
    inertia = []
    k_range = range(1, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_) 

    plt.plot(k_range, inertia, 'bx-')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Inertia')
    plt.title('Метод локтя')
    plt.show()

def k_means_method(data: pd.DataFrame, n_clusters: int):
    # Кластеризация с оптимальным k
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)

    # Добавляем кластеры к данным
    data = data.copy()
    data['cluster'] = clusters
    return data

def hierarchy_method(data: pd.DataFrame, n_clusters: int):
    clustering = AgglomerativeClustering(2, metric='euclidean', linkage='ward')
    lables = clustering.fit_predict(data)
    # 2. Для дендрограммы используем SciPy
    Z = linkage(data.values, method='ward')

    # Визуализация дендрограммы
    plt.figure(figsize=(10, 5))
    plt.title('Дендрограмма')
    plt.xlabel('Индекс данных')
    plt.ylabel('Расстояние')
    dendrogram(Z)
    plt.show()