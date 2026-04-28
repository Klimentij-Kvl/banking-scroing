import pandas as pd

def six_sigma_rule(df_column: pd.Series, return_bound: bool = True)->tuple[pd.Series, pd.Series, float, float]:
    mu = df_column.mean()
    sigma = df_column.std()
    lower_bound = mu - 6 * sigma
    upper_bound = mu + 6 * sigma
    lower_outliers = (df_column < lower_bound)
    upper_outliers = (df_column > upper_bound)
    if return_bound:
        return lower_outliers, upper_outliers, lower_bound, upper_bound
    return lower_outliers, upper_outliers

def delete_outliers(data: pd.DataFrame)->pd.DataFrame:
    df_filtered = data.copy()
    for col in df_filtered.columns:
        l_outliers, u_outliers = six_sigma_rule(data[col], return_bound=False)
        outliers = l_outliers | u_outliers
        df_filtered[col] = data[col][~outliers]
    return df_filtered

def censoring_data(data: pd.DataFrame)->pd.DataFrame:
    df_filtered = data.copy()
    for col in df_filtered.columns:
        l_outliers, u_outliers, l_bound, u_bound = six_sigma_rule(data[col])

        df_filtered.loc[l_outliers, col] = l_bound
        df_filtered.loc[u_outliers, col] = u_bound
    return df_filtered