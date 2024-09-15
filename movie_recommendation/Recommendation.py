import pandas as pd
import numpy as np
from math import sqrt

RATING_DATA_PATH = './data/ratings.csv' 

np.set_printoptions(precision=2)  # Output only to 2 decimal places

def distance(user_1, user_2):
    """Functions to calculate Euclidean distance"""
    return sqrt(np.sum((user_1 - user_2)**2))
    
    
def filter_users_without_movie(rating_data, movie_id):
    """Function to preemptively exclude users who haven't rated the 'movie_id'th movie"""
    return rating_data[~np.isnan(rating_data[:,movie_id])]
    
    
def fill_nan_with_user_mean(rating_data):
    """Fill empty values in rating data with the average value for each user"""
    filled_data = np.copy(rating_data)  
    row_mean = np.nanmean(filled_data, axis=1)  # Calculate user average rating
    
    inds = np.where(np.isnan(filled_data))  # Find empty indexes
    filled_data[inds] = np.take(row_mean, inds[0])  #Populate an empty index with user ratings
    
    return filled_data
    
    
def get_k_neighbors(user_id, rating_data, k):
    """Finds a user's neighbors for user_id"""
    distance_data = np.copy(rating_data)  
    # Add a column to hold the distance data
    distance_data = np.append(distance_data, np.zeros((distance_data.shape[0], 1)), axis=1)
    
    for i in range(len(distance_data)):
        row = distance_data[i]
        
        if i == user_id:  # Set the distance to infinity if they are the same user
            row[-1] = np.inf
        else:  # If you are a different user, store distance data in the last column
            row[-1] = distance(distance_data[user_id][:-1], row[:-1])
    
    # Sort data by distance column
    distance_data = distance_data[np.argsort(distance_data[:, -1])]
    
    # Returns only the k closest rows. + exclude the last (distance) column
    return distance_data[:k, :-1]
    
def predict_user_rating(rating_data, k, user_id, movie_id,):
    """Predict a user's movie rating based on a prediction matrix"""
    # Pre-exclude users who haven't watched 'movie_id' movie from the data
    filtered_data = filter_users_without_movie(rating_data, movie_id)
    # Create a new matrix filled with empty values
    filled_data = fill_nan_with_user_mean(filtered_data)
    # Find the data of k users similar to user user_id
    neighbors = get_k_neighbors(user_id, filled_data, k)

    return np.mean(neighbors[:, movie_id])
    
    
# test code
# Retrieve rating data
rating_data = pd.read_csv(RATING_DATA_PATH, index_col='user_id').values
# Predict User 0's rating for Movie 3 using 5 neighbors
print(predict_user_rating(rating_data, 5, 0, 3))