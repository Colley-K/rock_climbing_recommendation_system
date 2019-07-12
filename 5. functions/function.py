#All the functions I need to filter, sample, & graph my final capstone report

#imports

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from collections import defaultdict
import surprise as sp
from surprise.model_selection import cross_validate
from surprise.model_selection import KFold
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

##################################################################

#Basic counting & filtering Functions

#Matrix Density Function

def matrix_density(df):
    '''
    ---Parameters---
    df (Pandas DataFrame) of route, user, rating info

    ---Returns---
    -shape of the DF
    -number of unique routes and users
    -size of the matrix
    -density of the matrix
    '''

    x = df.route.nunique()
    y = df.user_id.nunique()
    print(f"Num of unique routes: {x}")
    print(f"Num of unique users: {y}")
    print(f"Matrix size: {x*y}")
    print(f"Shape of df: {df.shape}")
    print(f"Density of matrix: {(df.shape[0])/(x*y)}")

#Function to calculate the unique values

def unq_u_r(df):
    '''
    Return number of users and routes in df
    '''
    return df['user_id'].nunique(), df['route'].nunique()

#Counting number of users and routes per route or user rating

def ru_counts(df):
    '''
    For each climb, count number of users that rated it,
    and for each user, count number of climbs they rated
    ---Parameters---
    df: Pandas DataFrame
    ---Returns--
    u_r_counts: Pandas Series
    r_u_counts: Pandas Series
    '''
    u_r_counts = df.groupby(['user_id']).count()['route']
    r_u_counts = df.groupby(['route']).count()['user_id']
    return u_r_counts,r_u_counts

################################################################
#Functions for filtering and sampling!!!

#Function to filter out the entire DF based on user and route threshold

def filtering_df (df, u=0, r=0):
    '''
    Filters the data frame further based on computed cold-start thresholds

    ---Parameters---
    df (Pandas DataFrame) of route, user, rating info
    u (int) Number of ratings threshold for users
    r (int) Number of ratings threshold for routeIDs
    ---Returns---
    filtered_df (Pandas DataFrame)
    '''
    min_route_ratings = r
    filter_route = df['route'].value_counts() > min_route_ratings
    filter_route = filter_route[filter_route].index.tolist()

    min_user_ratings = u
    filter_user = df['user_id'].value_counts() > min_user_ratings
    filter_user = filter_user[filter_user].index.tolist()

    filtered_df = df[(df['route'].isin(filter_route)) & (df['user_id'].isin(filter_user))]

    print('The original data frame shape:\t{}'.format(df.shape))
    print('The new data frame shape:\t{}'.format(filtered_df.shape))
    return filtered_df


#Function for taking a percent sample of a data frame while keeping the itegreity of unique values in certain columns

def df_samp_unique_vals (df, percent, col1, col2=None):

    '''
    Takes a random sample of current dataframe while keeping a few column values unique to decrease matrix sparsity of sample

    ---Parameters---
    df (Pandas DataFrame)
    percent (float) enter a decimal of the percent sample you want
    col1 ("string") column name you want to keep retain unique values for (include quotation marks)
    col2 ("string") column name you want to keep retain unique values for (include quotation marks)

    ---Return---
    matrix stats of new df
    df_samp (Pandas DataFrame) as a percent sample of the original while keeping the columns entered unique
    '''

    #df.user_id.unique().sample(frac= percent) (more efficient code to explore??)
    df_drop = df.drop_duplicates(subset=[col1])
    print (f"User drop: {len(df_drop)}")
    if col2:
        df_drop = df_drop.drop_duplicates(subset=[col2])
        print (f"Route drop: {len(df_drop)}")
    #take a sample of the unique values
    sample1 = df_drop.sample(frac= percent, random_state=45)#Random state = random seed for .sample
    print (f"length of entire sample w/ unique users & routes: {len(sample1)}")

    #turn the unique routes & user names into a list to reference
    sample1= sample1.loc[:, [col1, col2]].values.T.ravel()
    lst1= sample1.tolist()

    #Filter out the original DF with only unique the unique values
    df_samp = df[(df[col1].isin(lst1)) & (df[col2].isin(lst1))]
    matrix_density(df_samp)
    return df_samp


################################################################
#Using testing & training models

#Creating Data objects from df's in surprise to later test & train
#Function to create surprise data objects from existing dataframes

def read_data_surprise (df, minstar= 1, maxstar=3, col1='user_id', col2= 'route', col3= 'rating'):
    '''
    Produces a surpise library data object from original dataframe

    ---Parameters---

    df (Pandas DataFrame)
    minstar (int) minimum star possible in dataset (default set to 1)
    maxstar (int) maximum star possible in dataset (default set to 3)
    col1 (string) column name that MUST correspond the the users in the df
    col2 (string) column name that MUST corresponds the the items in the df
    col3 (string) column name that corresponds the the ratings of the items in the df

    ---Returns---
    surprise library data object to manipulate later

    '''
    # need to specify the rating_scale of stars (default 1-3 stars)
    reader = sp.Reader(rating_scale=(minstar, maxstar))
    # The columns must correspond to user id, item id and ratings (in that order).
    data = sp.Dataset.load_from_df(df[[col1, col2, col3]], reader)

    return data

#Testing different algorithms
#Function to compute the RMSE of all the different alogithms offered int eh surprise library

def algo_tester(data_object):
  '''
  Produces a dataframe displaying all the different RMSE's, test & train times of the different surprise algorithms

  ---Parameters---
  data_object(variable) created from the read_data_surprise function

  ---Returns---
  returns a dataframe where you can compare the performance of different algorithms
  '''
  benchmark = []
  algos = [sp.SVDpp(), sp.SVD(), sp.SlopeOne(), sp.NMF(), sp.NormalPredictor(), sp.KNNBaseline(), sp.KNNBasic(), sp.KNNWithMeans(), sp.KNNWithZScore(), sp.BaselineOnly(), sp.CoClustering()]

  # Iterate over all algorithms
  for algorithm in algos:
    # Perform cross validation
    results = cross_validate(algorithm, data_object, measures=['RMSE'], cv=3, verbose=False)

    # Get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
    benchmark.append(tmp)

  benchmark= pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')
  return benchmark

#Using gridsearch to finetune algoirthm Parameters

def grid_search(data, param_grid, algorithm):
    '''
    inputs:
    data (surprise data object) made from initial DF
    param_grid (dictionary) which parameters to test out

    outputs:
    best RMSE(float) & combination of parameters (dictionary) that created that RMSE
    '''

    gs = GridSearchCV(algorithm, param_grid, measures=['rmse'], cv=3)
    gs.fit(data)

    # best RMSE score
    print(gs.best_score['rmse'])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params['rmse'])

###########################################################################################

#PREDICTIONS FUNCTIONS

#Prediction function- returns the errors of the prediction
def get_predict_df (predictions):

    #Creating a dataframe with the predictions
    df_predict = pd.DataFrame(predictions, columns=['uid', 'iid', 'rui', 'est', 'details'])
    df_predict['err'] = abs(df_predict.est - df_predict.rui)

    #best predictions
    best_predictions = df_predict.sort_values(by='err')[:10]
    print ("Best Predictions:")
    print (best_predictions)

    #worst predictions
    worst_predictions = df_predict.sort_values(by='err')[-10:]
    print ("Worst Predictions:")
    print (worst_predictions)

    #Mean & Std of error column
    mean_err = df_predict['err'].mean()
    std_err = df_predict['err'].std()
    print (f"Average error: {mean_err}")
    print (f"Standard deviation of error: {std_err}")

    #percent errors
    err2 =df_predict[df_predict['err'] > 2]
    print (f"There are approxmiately {(len(err2)/(len(df_predict))*100):.5f} percent of predictions with a 2 star error or greater")
    err1_5 =df_predict[df_predict['err'] > 1.5]
    print (f"There are approxmiately {(len(err1_5)/(len(df_predict))*100)} percent of predictions with a 1.5 star error or greater")
    err1 =df_predict[df_predict['err'] > 1]
    print (f"There are approxmiately {(len(err1)/(len(df_predict))*100)} percent of predictions with a 1 star error or greater")
    err_5 =df_predict[df_predict['err'] > .5]
    print (f"There are approxmiately {(len(err_5)/(len(df_predict))*100)} percent of predictions with a .5 star error or greater")

    return df_predict

#Top 10 dataframe algorithm
def get_top_n(predictions, n=10):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n




#################################################################

#Functions for graphs!

#Violin plot of violin_accuracy

def violin_accuracy(algo, testset):
    '''
    Plot violin chart showing descrepancy between true and predicted ratings with means & standard deviations included

    ---Parameters---
    algo (predefined variable) of the algorithm being used
    testset (variable) of which test data set being used

    ---Returns---
    Means and standard deviations of the estimated predictions per true rating
    Violin chart comparing real ratings to predicted ratings

    '''

    predictions = algo.test(testset)
    predict_df = pd.DataFrame(predictions)

    #Calculating Means and standard deviations per true rating
    p_mean= predict_df.groupby("r_ui")["est"].mean()
    p_std = predict_df.groupby("r_ui")["est"].std()
    print (type(p_mean))
    print (f"Means of Predicted per True Rating: {p_mean}")
    print (f"STD of Predicted per True Rating: {p_std}")

    #Plotting the Violin plot
    sns.violinplot( x=predict_df["r_ui"], y= predict_df["est"], saturation= .5)
    plt.xlabel('True Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('True vs. Predicted Ratings Using SVD')


#Histogram of Average Star Count per column selected

def hist_avg_star (df, col1, color= None):
    '''
    Creates a Histogram based on the average star ratings of a particular column

    ---Parameters---
    df (Pandas DataFrame)
    col1 ("string") column name for aggregation in string format including quotation marks
    color ("string") color from matplot lib to customize color of histogram in string format including quotation marks

    ---Returns---
    histogram
    '''
    if color:
        plt.hist(df.groupby(col1).mean()['rating'], bins = 10, rwidth=.85, color= color, alpha= .45)
    else:
        plt.hist(df.groupby(col1).mean()['rating'], bins = 10, rwidth=.85)
    plt.xlabel('Star Value')
    plt.ylabel('Counts')
    plt.title(f'Count of {col1} by average star value', size=16)
    plt.show()

#Function create a scatter plot out of the star ratings

def count_ratings(df, column, cmap= None):
    '''
    Creates a scatter plot based on the average star ratings of a particular column

    ---Parameters---
    df (Pandas DataFrame)
    column ("string") column name for aggregation in string format including quotation marks
    cmap ("string") color map from matplot lib to customize color of scatter plot in string format including quotation marks

    ---Returns---
    scatter plot
    '''

    df_agg = df.groupby(column).agg(['mean','count'])['rating']
    df_agg['meanxcount'] = (df_agg['mean'] * np.log(np.log(df_agg['count'])))**2.5
    plt.scatter(df_agg['mean'], df_agg['count'],alpha = 1, cmap=cmap,c=df_agg['meanxcount'])
    plt.xlabel('Star Value')
    plt.ylabel('Counts')
    plt.ylim(0,750)
    plt.title(f'Count of {column} by Average Star Value', size=16)
    plt.colorbar(ticks=[0])
    plt.show()

#Creating count histograms

def plt_count_hists(df, n=30):
    '''
    Plot hists of rating counts, routes by user and users by route
    ---Parameters---
    df (Pandas DataFrame) RUS DataFrame
    n (int) limit x axis and bin everything to the right
    '''
    u_r_counts,r_u_counts = ru_counts(df)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(14,5))
    ax1.hist(u_r_counts, color= "#7a0177", bins=list(range(1,n+1)))
    ax1.set_title('Count user ratings by route')
    ax1.set_ylabel('User count')
    ax1.set_xlabel('Number of user ratings')
    ax2.hist(r_u_counts, color= "#f768a1", bins=list(range(1,n+1)))
    ax2.set_title('Count route ratings by user')
    ax2.set_ylabel('Route count')
    ax2.set_xlabel('Number of route ratings')
    plt.show()

#Function for the Cumulative Sum of ratings per user with unfiltered data

def plt_cumsum(df, n=30):
    '''
    Plot proportion of users excluded by varying threshold
    ---Parameters---
    df (Pandas DataFrame) RUS DataFrame
    n (int) limit x axis and bin everything to the right
    '''
    u_r_counts, r_u_counts = ru_counts(df)
    u_r_cumsum = u_r_counts.value_counts().sort_index().cumsum() / unq_u_r(df)[0]
    r_u_cumsum = r_u_counts.value_counts().sort_index().cumsum() / unq_u_r(df)[1]
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(14,5))
    ax1.plot(u_r_cumsum, color= "b")
    ax1.set_xlim([0, n+1])
    ax1.set_title('Cumsum user ratings by route')
    ax1.set_ylabel('Proportion of users excluded')
    ax1.set_xlabel('Number of user ratings - thresh')
    ax2.plot(r_u_cumsum, color= "b")
    ax2.set_xlim([0, n+1])
    ax2.set_title('Cumsum route ratings by user')
    ax2.set_ylabel('Proportion of routes excluded')
    ax2.set_xlabel('Number of route ratings - thresh')
    plt.show()
