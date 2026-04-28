from typing import Tuple
import pandas as pd
import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.preprocessing import MinMaxScaler
#from factor_analyzer import FactorAnalyzer
#from factor_analyzer 


def bartlett_test_of_sphericity(n: int, p: int, corr_matr):
    det_R = np.linalg.det(corr_matr)
    chi_square = -((n-1) - (2*p+5)/6) * np.log(det_R)
    df = p * (p-1) // 2 
    p_value = 1 - chi2.cdf(chi_square, df)
    return p_value

"""
def pca_method(data: pd.DataFrame, n_components: int):
    # pca = PCA(n_components=n_components)
    # principal_components = pca.fit_transform(data)

    # scaler = MinMaxScaler()
    # pca_normalized = scaler.fit_transform(principal_components)

    fa = FactorAnalyzer(n_factors=n_components, rotation=None)
    fa.fit(data)
    scores = fa.transform(data)
    loadings = fa.loadings_

    scaler = MinMaxScaler()
    pca_norm = scaler.fit_transform(scores)

    s = []
    for i in range(n_components):
        s.append('PC' + str(i+1))
    pca_df = pd.DataFrame(data=pca_norm, columns=s)

    loadings = pd.DataFrame(loadings, columns=s, index=data.columns)

    return pca_df,loadings
"""

def pca_method2(data: pd.DataFrame, rotate: bool = True, rotation_method: str= 'varimax'):
    observation_names = data.index.tolist()
    feature_names = data.columns.tolist()
    n_samples, n_features = data.shape

    #нормировка данных
    means = data.mean().values
    stds = data.std().values
    #Z = (data.values - means) / stds
    Z = data.values

    #матрица корреляции
    R = np.corrcoef(Z.T)

    #собственные векторы и значения
    eigenvalues, eigenvectors = np.linalg.eigh(R)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    #матрица нагрузок
    loadings = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    #главные компоненты
    scores = Z @ eigenvectors

    #вычисляем долю объяснённой дисперсии
    total_variance = eigenvalues.sum()
    explained_variance = eigenvalues / total_variance
    cumulative_variance = np.cumsum(explained_variance)

    #возьмем лишь те компоненты, с.зн. которых больше единицы
    n_components = np.sum(eigenvalues > 1)

    #оставляем только главные компоненты
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]
    loadings = loadings[:, :n_components]
    scores = scores[:, :n_components]
    explained_variance = explained_variance[:n_components]
    cumulative_variance = cumulative_variance[:n_components]

    # 5. Применяем квартимакс вращение
    #rotator = QuartimaxRotation(normalize=True)
    rotated_loadings = quartimax(loadings)
    
    # 6. Вычисляем повернутые компоненты и собственные значения
    rotated_components = rotated_loadings / np.sqrt(eigenvalues)
    
    # 7. Вычисляем дисперсии после вращения
    # После вращения дисперсии могут измениться
    explained_variance_rotated = np.sum(rotated_loadings ** 2, axis=0)
    
    # 8. Вычисляем новые счета (scores)
    scores = Z @ rotated_components

    #интегральный показатель
    weights = eigenvalues / eigenvalues.sum()
    integral_index = scores @ weights

    pc_names = [f'PC{i}' for i in range(1,n_components+1)]
    scores_df = pd.DataFrame(scores, index=observation_names, columns=pc_names)
    loadings_df = pd.DataFrame(loadings, index=feature_names, columns=pc_names)
    rotated_loadings_df = pd.DataFrame(rotated_loadings, index=feature_names, columns=pc_names)
    """
    eigenvalues_df = pd.DataFrame({
        'Собственные значения': eigenvalues,
        'Доля дисперсии': explained_variance,
        'Кумулятивная доля': cumulative_variance
    }, index=pc_names)
    
    # 4. Интегральный показатель
    integral_df = pd.DataFrame({
        'Интегральный показатель': integral_index
    }, index=observation_names)

    return {
        'scores': scores_df,
        'loadings': loadings_df,
        'eigenvalues': eigenvalues_df,
        'explained_variance': eigenvalues_df['Доля дисперсии'].values,
        'cumulative_variance': eigenvalues_df['Кумулятивная доля'].values,
        'integral_index': integral_df,
        'original_data': data,
    }
    """
    return {
        'original_loadings': loadings_df,
        'rotated_loadings': rotated_loadings_df,
        #'original_components': components,
        'rotated_components': rotated_components,
        'original_eigenvalues': eigenvalues,
        'rotated_eigenvalues': explained_variance_rotated,
        'scores': scores,
        'explained_variance_ratio': eigenvalues / np.sum(eigenvalues),
        'explained_variance_ratio_rotated': explained_variance_rotated / np.sum(explained_variance_rotated)
    }

def quartimax(A: np.ndarray) -> np.ndarray:
    """
    Выполняет квартимакс вращение матрицы нагрузок.
    
    Parameters:
    -----------
    A : np.ndarray
        Исходная матрица факторных нагрузок (n_features × n_factors)
        
    Returns:
    --------
    np.ndarray : повернутая матрица нагрузок
    """
    if A.ndim != 2:
        raise ValueError("Матрица нагрузок должна быть 2D")
        
    p, k = A.shape
    # Сохраняем исходную матрицу для денормализации
    A_original = A.copy()
    
    # 1. Нормализация Kaiser (если требуется)
    communalities = np.sum(A ** 2, axis=1)
    # Защита от нулевых коммунальностей
    communalities = np.maximum(communalities, 1e-10)
    D = np.diag(1.0 / np.sqrt(communalities))
    A_norm = D @ A
    
    # 2. Инициализация матрицы вращения
    T = np.eye(k)
    A_rotated = A_norm.copy()
    
    # 3. Итеративная оптимизация
    converged = False
    Q_old = quartimax_criterion(A_rotated)
    
    for i in range(100):
        # Для каждого уникального сочетания факторов
        for i_factor in range(k):
            for j_factor in range(i_factor + 1, k):
                # Выделяем пару факторов для вращения
                Ai = A_rotated[:, i_factor]
                Aj = A_rotated[:, j_factor]
                
                # Вычисляем углы вращения
                num = np.sum(Ai * Aj * (Ai**2 - Aj**2))
                denom = np.sum(Ai**4 - Aj**4 - (Ai**2 - Aj**2)**2)
                
                # Защита от деления на ноль
                if np.abs(denom) < 1e-12:
                    phi = 0
                else:
                    phi = 0.25 * np.arctan2(num, denom)
                
                # Создаем матрицу вращения для пары факторов
                G = np.eye(k)
                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)
                G[i_factor, i_factor] = cos_phi
                G[i_factor, j_factor] = -sin_phi
                G[j_factor, i_factor] = sin_phi
                G[j_factor, j_factor] = cos_phi
                
                # Применяем вращение
                A_rotated = A_rotated @ G.T
                T = T @ G.T
        
        # Проверка сходимости
        Q_new = quartimax_criterion(A_rotated)
        if np.abs(Q_new - Q_old) < 1e-8:
            converged = True
            break
        Q_old = Q_new
    
    #self.n_iter_ = i if converged else self.max_iter
    #self.rotation_matrix_ = T
    
    # 4. Денормализация (возвращаем к исходному масштабу)
    A_final = np.linalg.inv(D) @ A_rotated
    
    return A_final

def quartimax_criterion(A: np.ndarray) -> float:
    """
    Критерий квартимакс: максимизация суммы квадратов нагрузок фактора.
    
    Q = Σ_i Σ_j a_ij^4 - (1/p) * (Σ_i Σ_j a_ij^2)^2
    
    Parameters:
    -----------
    A : np.ndarray
        Матрица факторных нагрузок (n_features × n_factors)
        
    Returns:
    --------
    float : значение критерия квартимакс
    """
    p, k = A.shape
    sum_sq = np.sum(A ** 2)
    sum_quad = np.sum(A ** 4)
    
    # Упрощенный критерий квартимакс (без нормализации)
    Q = sum_quad - (1/p) * sum_sq ** 2
    return Q