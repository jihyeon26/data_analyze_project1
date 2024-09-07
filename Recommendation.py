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
    
    # 가장 가까운 k개의 행만 리턴한다 + 마지막(거리) 열은 제외한다
    return distance_data[:k, :-1]
    
def predict_user_rating(rating_data, k, user_id, movie_id,):
    """예측 행렬에 따라 유저의 영화 평점 예측 값 구하기"""
    # movie_id 번째 영화를 보지 않은 유저를 데이터에서 미리 제외시킨다
    filtered_data = filter_users_without_movie(rating_data, movie_id)
    # 빈값들이 채워진 새로운 행렬을 만든다
    filled_data = fill_nan_with_user_mean(filtered_data)
    # 유저 user_id와 비슷한 k개의 유저 데이터를 찾는다
    neighbors = get_k_neighbors(user_id, filled_data, k)

    return np.mean(neighbors[:, movie_id])
    
    
# 테스트 코드
# 평점 데이터를 불러온다
rating_data = pd.read_csv(RATING_DATA_PATH, index_col='user_id').values
# 5개의 이웃들을 써서 유저 0의 영화 3에 대한 예측
print(predict_user_rating(rating_data, 5, 0, 3))