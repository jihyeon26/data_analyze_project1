import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RATING_DATA_PATH = 'movie_recommendation_matrix/data/ratings.csv'  

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

def predict(Theta, X):
    #A function that calculates the predicted value by multiplying user preferences and product attributes.
    return Theta @ X


def cost(prediction, R):
    #Function that calculates the loss of the matrix factorization algorithm
    return np.nansum((prediction - R)**2)


def initialize(R, num_features):
    #A function that randomly creates user taste and product attribute matrices.
    num_users, num_items = R.shape
    
    Theta = np.random.rand(num_users, num_features)
    X = np.random.rand(num_features, num_items)
    
    return Theta, X


def gradient_descent(R, Theta, X, iteration, alpha, lambda_):
    #Matrix factorization gradient descent function
    num_user, num_items = R.shape
    num_features = len(X)
    costs = []
        
    for _ in range(iteration):
        prediction = predict(Theta, X)
        error = prediction - R
        costs.append(cost(prediction, R))
                          
        for i in range(num_user):
            for j in range(num_items):
                if not np.isnan(R[i][j]):
                    for k in range(num_features):
                        
                        Theta[i][k] -= alpha * (np.nansum(error[i, :] * X[k, :]) + lambda_*Theta[i][k])
                        X[k][j] -= alpha * (np.nansum(error[:, j] * Theta[:, k]) + lambda_*X[k][j])
                       
    return Theta, X, costs


# --------------test code----------------

# Read data
ratings_df = pd.read_csv(RATING_DATA_PATH, index_col='user_id')

# Apply mean normalization to rating data
for row in ratings_df.values:
    row -= np.nanmean(row)
       
R = ratings_df.values
        
Theta, X = initialize(R, 5)  # Initialize matrix
Theta, X, costs = gradient_descent(R, Theta, X, 200, 0.001, 0.01)  # gradient descent
    


print(Theta)
print(X)