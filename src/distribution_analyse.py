import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
import seaborn as sns

def analyse_column(df: pd.DataFrame, col: str):
    print(f"Колонка: {col}")
    print("Минимум:", df[col].min(), end='; ')
    print("Максимум:", df[col].max(), end='; ')
    print("Среднее:", df[col].mean(), end='; ')
    print("Медиана:", df[col].median(), end='; ')
    print("Стандартное отклонение:", df[col].std(), end='\n')
    data = df[col].dropna()
    counts, bins, patches = plt.hist(data, bins=100, color='lightblue', edgecolor='black', alpha=0.7)
    mu = data.mean()
    sigma = data.std()
    x = np.linspace(data.min(), data.max(), 200)
    y = norm.pdf(x, mu, sigma)
    y_scaled = y * max(counts) / max(y)
    plt.plot(x, y_scaled, 'r-', linewidth=2)
    plt.show()

def show_box_plot_of_column(df_column: pd.Series)->None:
    sns.boxplot(x=df_column)
    plt.show()

def analyse_data(df: pd.DataFrame):
    analyse_column(df, 'k1')