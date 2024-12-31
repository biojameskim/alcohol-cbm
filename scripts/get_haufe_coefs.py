import numpy as np
from sklearn.preprocessing import StandardScaler

def get_haufe_coefs(matrix_type, file_name, sex):
    """
    [get_haufe_coefs] calculates the Haufe coefficients for a given set of coefficients and input data.
    First the average coefficients are calculated (over the number of iterations and then transformed)
    Source: (Haufe et. al 2014) https://www.sciencedirect.com/science/article/pii/S1053811913010914#s0210
    """
    coefs = np.load(f'results/{matrix_type}/logreg/{file_name}/logreg_{matrix_type}_{file_name}{sex}_coefficients.npy')
    X = np.load(f'data/training_data/aligned/X_{matrix_type}_{file_name}{sex}.npy')
    X = StandardScaler().fit_transform(X) # Standardize the data

    coefs = np.mean(coefs, axis=0) # Average the coefficients over the number of iterations
    cov_X = np.cov(X, rowvar=False) # Covariance matrix of X (cols are features so rowvar=False)
    return cov_X @ coefs