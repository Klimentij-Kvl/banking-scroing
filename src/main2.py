import numpy as np
import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn_imports

df, meta = pyreadstat.read_sav('data.sav')
df = df.drop('k4_new', axis=1)
selected_columns = df.columns[1:-1]

df = sklearn_imports.censor_data(df, selected_columns)
df = sklearn_imports.normalize_data(df, selected_columns)
corr_matr, df, selected_columns = sklearn_imports.delete_strong_correlations(df, selected_columns)

bartlett_chi_square, bartlett_pvalue = sklearn_imports.bartlett_test_of_sphericity(df, selected_columns)
kmo_pvalue = sklearn_imports.kmo_test(df, selected_columns)
#pca_dict, data_pca, weight = sklearn_imports.principal_component_algorithm(df, selected_columns, 6)
pca_rotated_dict, data_pca_rotated = sklearn_imports.principal_component_algorithm(df, selected_columns, 6,
                                                                  rotation='varimax', method='minres')
rotated_eigenvalues = np.sum(pca_rotated_dict['loadings']**2, axis=0)

kmeans_data = sklearn_imports.kmeans_method(data_pca_rotated[:-1], 4)

hierarchy_data = sklearn_imports.hierarchy_method(data_pca_rotated[0:-1], 4, show_grahpic=True)

"""
plt.figure(figsize=(10, 7))
sns.scatterplot(data=hierarchy_data, x='PC1', y='PC2', hue='label', palette='Set2', s=100)
plt.title('Распределение объектов по кластерам')
plt.legend(title='Кластеры')
plt.show()
"""