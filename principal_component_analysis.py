import numpy as np
import pandas as pd

class SVDPCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components

    def fit(self, X: pd.DataFrame)->None:
        self.corr_matrix = X.corr()
        self.U, self.S, self.Vt = np.linalg.svd(self.corr_matrix)
        print(self.S)