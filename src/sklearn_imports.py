import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

def normalize_data(data: pd.DataFrame, selected_columns: list):
    data_scaled = data.copy()
    scaler = StandardScaler()
    data_scaled[selected_columns] = scaler.fit_transform(data[selected_columns])
    return data_scaled

def ee_algorithm(data: pd.DataFrame, selected_columns: list):
    ee_method = EllipticEnvelope(random_state=30, contamination=0.1)
    labels = ee_method.fit_predict(data[selected_columns])
    return labels

def censor_data(data: pd.DataFrame, selected_columns: list):
    labels = ee_algorithm(data, selected_columns)
    data_clean = data[labels != -1].copy()
    return data_clean

def show_correlation_heatmap(data: pd.DataFrame, selected_columns: list):
    corr_matr = data[selected_columns].corr().abs()
    sns.heatmap(corr_matr)
    plt.show()

def delete_strong_correlations(data: pd.DataFrame, selected_columns: list):
    data_clean = data[selected_columns].copy()
    corr_matr = data[selected_columns].corr()
    corr_matr_values = corr_matr.values
    for i in range(corr_matr.shape[0]):
        for j in range(i):
            if np.abs(corr_matr_values[i,j]) > 0.95:
                col_for_deleting = corr_matr.columns[i]
                data_clean = data_clean.drop(col_for_deleting, axis=1)
    return corr_matr, data_clean, data_clean.columns

def bartlett_test_of_sphericity(data: pd.DataFrame, selected_columns: list):
    chi_square_value, p_value = calculate_bartlett_sphericity(data[selected_columns])
    return chi_square_value, p_value

def kmo_test(data: pd.DataFrame, selected_columns: list = None):
    _, kmo_model = calculate_kmo(data[selected_columns])
    return kmo_model

def principal_component_algorithm(data: pd.DataFrame, selected_columns: list, n_components: int,
                                   rotation: str = None, method: str = 'minres'):
    fa = FactorAnalyzer(n_factors=n_components, rotation=rotation, method=method)
    fa.fit(data[selected_columns])
    pc_columns=[f'PC{i+1}' for i in range(n_components)]
    pc_index = selected_columns

    eigenvalues = fa.get_eigenvalues()[0]

    loadings = fa.loadings_
    loadings_df = pd.DataFrame(loadings, index=pc_index, columns=pc_columns)

    variance = fa.get_factor_variance()
    variance_df = pd.DataFrame({
        'Фактор': pc_columns,
        'SS Loadings': variance[0],
        'Proportion Var': variance[1],
        'Cumulative Var': variance[2]
    })

    factor_scores = fa.transform(data[selected_columns])
    data_pca = pd.DataFrame(factor_scores, columns=pc_columns, index=data.index)
    
    integral_score = np.zeros(len(data_pca))
    for i in range(n_components):
        integral_score += factor_scores[:,i] * variance[1][i]

    data_pca['integral_score'] = integral_score

    results = {
        'loadings': loadings_df,
        'variance': variance_df,
        'eigenvalues': eigenvalues
    }

    return results, data_pca

def kmeans_method(data: pd.DataFrame, n_clusters: int):
    clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = clustering.fit_predict(data)

    data_labels = data.copy()
    data_labels['label'] = labels
    return data_labels

def hierarchy_method(data: pd.DataFrame, n_clusters: int, show_grahpic: bool = False):
    clustering = AgglomerativeClustering(n_clusters, metric='euclidean', linkage='ward')
    labels = clustering.fit_predict(data)
    # 2. Для дендрограммы используем SciPy
    Z = linkage(data.values, method='ward')

    if show_grahpic:
        # Визуализация дендрограммы
        plt.figure(figsize=(10, 5))
        plt.title('Дендрограмма')
        plt.xlabel('Индекс данных')
        plt.ylabel('Расстояние')
        dendrogram(Z)
        plt.show()
    data_labels = data.copy()
    data_labels['label'] = labels

    return data_labels
