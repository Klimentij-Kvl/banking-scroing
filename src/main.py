import numpy as np
import pandas as pd
import pyreadstat
import matplotlib.pyplot as plt
import seaborn as sns

import ee_algorithm
import principal_component_method
import cluster_analysis
import math

df, meta = pyreadstat.read_sav('data.sav')
df_prepared, meta2 = pyreadstat.read_sav('prepared_data.sav')
df = df.drop('k4_new', axis=1)
df_prepared = df_prepared.drop('k4_new_n', axis=1)

selected_columns = df.columns[1:-1]
prepared_selected_columns = df_prepared.columns[-14:]

df_clean = ee_algorithm.prepare_data(df, selected_columns)
corr_matr=df_clean[selected_columns].corr().abs()
df_clean = df_clean.drop('k8', axis=1) #корреляционный показал, что между k8 и k5 корреляция практически равна 1

selected_columns_clean = df_clean.columns[1:-1]
pca_selected_columns = ['k1','k2','k4','k5','k6','k7','k9','k10','k11','k13','k14','k15','k18','k19']
#df_pca, loadings = principal_component_method.pca_method(df_prepared[prepared_selected_columns], 5)
#print(loadings[np.abs(loadings) > 0.41])

corr_matr2 = df_clean[selected_columns_clean].corr().abs()
#print(principal_component_method.bartlett_test_of_sphericity(len(df_clean['k1']), 19, corr_matr2))

#cluster_analysis.elbow_method(df_prepared[prepared_selected_columns])
df_clusters = cluster_analysis.k_means_method(df_prepared[prepared_selected_columns], 4)
#df_clusters = cluster_analysis.k_means_method(df_pca, 4)
cluster_counts = df_clusters['cluster'].value_counts()
cluster_perc = cluster_counts / len(df_clusters['cluster'])
cluster_perc.index.name = "cluster percentage"
print(cluster_counts)
print(cluster_perc)


dict = principal_component_method.pca_method2(df_clean[selected_columns_clean], rotate=True)
loadings_df = dict['original_loadings']
#print(loadings_df[loadings_df.abs() > 0.4])
rotated_loadings_df = dict['rotated_loadings']
print(rotated_loadings_df[rotated_loadings_df.abs() > 0.4])

df_clusters = cluster_analysis.k_means_method(df_clean[pca_selected_columns], 4)
#df_clusters = cluster_analysis.k_means_method(df_pca, 4)
cluster_counts = df_clusters['cluster'].value_counts()
cluster_perc = cluster_counts / len(df_clusters['cluster'])
cluster_perc.index.name = "cluster percentage"
print(cluster_counts)

df_clusters = cluster_analysis.hierarchy_method(df_clean[pca_selected_columns], 4)
#df_clusters = cluster_analysis.k_means_method(df_pca, 4)
cluster_counts = df_clusters['cluster'].value_counts()
cluster_perc = cluster_counts / len(df_clusters['cluster'])
cluster_perc.index.name = "cluster percentage"
print(cluster_counts)

#print(loadings_df[np.abs(loadings_df) > 0.41])
#print(dict['eigenvalues'])
#print(cluster_counts)
#print(cluster_perc)

#print(df_pca)
#sns.heatmap(df_pca.corr())

#print(len(df), len(df_prepared), len(df_clean))
#sns.histplot(df_clean, x='k2', color='green')
#sns.histplot(df_prepared, x='k2_n', color='red')
#sns.histplot(df_pca, x='k2_n',color='blue')
#sns.heatmap(corr_matr)

plt.show()