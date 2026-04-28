import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def filter_six_sigma_rule(df_column: pd.Series)->tuple[float,float,float]:
    mu = df_column.mean()
    sigma = df_column.std()
    lower_bound = mu - 6 * sigma
    upper_bound = mu + 6 * sigma
    filtered_column = df_column[(df_column >= lower_bound) & (df_column <= upper_bound)]
    return lower_bound, upper_bound, (len(df_column)-len(filtered_column))/len(df_column)*100

def filter_hampel_test(df_column: pd.Series)->tuple[float, float, float]:
    x_med = df_column.median()
    x_mad = (df_column - x_med).abs().median()
    k = 10
    lower_bound = x_med - k * x_mad
    upper_bound = x_med + k * x_mad
    
    filtered_column = df_column[(df_column >= lower_bound) & (df_column <= upper_bound)]
    #print(len(df_column), len(filtered_column))
    return lower_bound, upper_bound, (len(df_column)-len(filtered_column))/len(df_column)*100

def filter_extreme_observations(df: pd.DataFrame)->tuple[pd.DataFrame, float]:
    df_filtered = df.copy()
    filtered_percente = 0
    for col in df_filtered.columns[1:-1]:
        print(col,end=' ')
        lower_bound, upper_bound, buff = filter_six_sigma_rule(df[col])
        print(buff)
        df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]
        #print(df_filtered)
        filtered_percente = (filtered_percente + buff)/2
    return df_filtered, filtered_percente

def test(df: pd.DataFrame):
    df_filtered, percents = filter_extreme_observations(df)

    fig, axes = plt.subplots(1, 2, figsize=(10,4))
    sns.boxplot(y=df['k2'], ax=axes[0], color='lightcoral')
    axes[0].set_title("До очистки (включая выбросы)")
    sns.boxplot(y=df_filtered['k2'], ax=axes[1], color='lightgreen')
    axes[1].set_title("После очистки при помощи теста Хампела")
    plt.tight_layout()

    #print(df.shape, df_filtered.shape)
    print("Процент фильтрации:", percents)
    plt.show()