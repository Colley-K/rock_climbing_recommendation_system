#Functions to create the coldstart threshold RMSE/ SURVIVORSHIP graphs/analysis

#Imports

#Installs
#pip install surprise

#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import base64
from collections import defaultdict
import surprise as sp
from surprise.model_selection import cross_validate, KFold, train_test_split


def algo_metrics(df):
    '''
    Return metrics algo metrics for df: rmse
    ---Parameters---
    df (Pandas DataFrame) RUS DataFrame
    u (int) Number of ratings threshold for users
    r (int) Number of ratings threshold for routeIDs
    ---Returns---
    RMSE metrics
    '''
    reader = sp.Reader(line_format='user item rating', sep=',', skip_lines=1)
    data = sp.Dataset.load_from_df(df, reader=reader)
    trainset, testset = train_test_split(data, test_size=.2)

    # Fit out of the box SVD to trainset and predict on test set
    algo = sp.SVD()
    algo.fit(trainset)
    predictions = algo.test(testset)
    return sp.accuracy.rmse(predictions)

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

def df_add_counts(df):
    '''
    Add columns for counts to df:
    - user ratings per route
    - route ratings per user
    ---Parameters---
    df (Pandas DataFrame) RUS DataFrame
    u (int) Number of ratings threshold for users
    r (int) Number of ratings threshold for routeIDs
    ---Returns---
    df_counts (Pandas DataFrame)
    '''
    # count ratings per user and route
    u_r_counts,r_u_counts = ru_counts(df)
    df_r_ucounts = r_u_counts.reset_index()
    df_r_ucounts.columns = ['route','count_ru']
    df_u_rcounts = u_r_counts.reset_index()
    df_u_rcounts.columns = ['user_id','count_ur']

    # merge ratings back onto original df
    df_counts = pd.merge(df, df_u_rcounts, on='user_id')
    df_counts = pd.merge(df_counts, df_r_ucounts, on='route')
    return df_counts

def add_counts_bins(df):
    '''
    Add columns for counts to df:
    - user ratings per route
    - route ratings per user
    - user bins based on number of routes rated
    ---Parameters---
    df (Pandas DataFrame) RUS DataFrame
    u (int) Number of ratings threshold for users
    r (int) Number of ratings threshold for routeIDs
    ---Returns---
    df_counts_bins (Pandas DataFrame)
    '''
    df_counts_bins = df_add_counts(df)
    conditions = [  (df_counts_bins['count_ur'] <= 5),
                    (df_counts_bins['count_ur'] >  5) & (df_counts_bins['count_ur'] <= 10),
                    (df_counts_bins['count_ur'] > 10) & (df_counts_bins['count_ur'] <= 15),
                    (df_counts_bins['count_ur'] > 15) & (df_counts_bins['count_ur'] <= 20),
                    (df_counts_bins['count_ur'] > 20)]
    choices = ['u_bin_0_5', 'u_bin_6_10', 'u_bin_11_15', 'u_bin_16_20', 'u_bin_20+']
    df_counts_bins['u_bin'] = np.select(conditions, choices, default='None')
    return df_counts_bins

def rus_chop(df, u=5, r=5):
    '''
    Chop RUS DF based on minimum cold-start thresholds.
    ---Parameters---
    df (Pandas DataFrame) RUS DataFrame
    u (int) Number of ratings threshold for users
    r (int) Number of ratings threshold for routeIDs
    ---Returns---
    df_chopped (Pandas DataFrame) Chopped RUS DataFrame
    '''
    # count ratings per user and route
    df_counts = df_add_counts(df)

    # chop df based on threshold
    df_chopped = df_counts[(df_counts['count_ur'] >= u) & (df_counts['count_ru'] >= r)]
    return df_chopped[['user_id','route','rating']]

def thresh_metrics_arrs(df, u=5, r=5):
    '''
    Compute metrics for every combination of thresholds
    ---Parameters---
    df (Pandas DataFrame) RUS DataFrame
    u (int) Number of ratings threshold for users
    r (int) Number of ratings threshold for routeIDs
    ---Returns---
    arr_rmse (np-array)
    '''
    arr_rmse = np.zeros((u,r))

    for i in range(u):
        for j in range(r):
            df_chopped = rus_chop(df,i,j)
            rmse = algo_metrics(df_chopped)
            arr_rmse[i,j] = rmse

    return arr_rmse

def plot_RMSE_userthresh(df, arr_rmse, title, r=0):
    '''
    Plot RMSE by user-coldstart threshold
    assumes route threshold = r
    '''

    a = np.array(arr_rmse)
    print (np.mean(a, axis=1))
    print (np.mean(a, axis=0))

    arr_rmse_df = pd.DataFrame(arr_rmse)
    usersurvivors = []
    for u in range(20):
        usersurvivors.append(unq_u_r(rus_chop(df,u,r))[0])
    usersurvivors_df = pd.DataFrame(usersurvivors,columns=['survivors']).transpose()
    arr_rmse_survivors_df = pd.concat([arr_rmse_df,usersurvivors_df],axis=0).transpose()

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(arr_rmse_survivors_df[0],c='blue',label='RMSE')

    ax.set_title(title, size=18)
    ax.set_ylim(.62,.64)
    ax.set_xlabel('Cold-start threshold')
    ax.set_ylabel('RMSE')
    ax.axvline(x=5, c="r")
    ax.axvline(x=13, c="r")

    ax2 = ax.twinx()
    ax2.plot(arr_rmse_survivors_df['survivors'],c='green',label='Users Retained')
    ax2.set_ylim(0,25000)
    ax2.set_ylabel('Users retained')
    ax2.grid(False)

    #Combining both lines into one legend
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc='upper right')
    plt.show()

def plot_RMSE_routethresh(df, arr_rmse, title, u=0):
    '''
    Plot RMSE by route-coldstart threshold; assumes user threshold = u
    '''
    arr_rmse_df = pd.DataFrame(arr_rmse)
    route_survivors = []
    for r in range(20):
        route_survivors.append(unq_u_r(rus_chop(df,u,r))[1])
    route_survivors_df = pd.DataFrame(route_survivors,columns=['survivors'])
    arr_rmse_survivors_df = pd.concat([arr_rmse_df,route_survivors_df],axis=1)
    print (arr_rmse_survivors_df)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(arr_rmse_survivors_df[0],c='blue',label='RMSE')
    ax.set_title(title, size=16)
    ax.set_ylim(.62,.64)
    ax.set_xlabel('Cold-start threshold')
    ax.set_ylabel('RMSE')
    ax.axvline(x=12, c="r")

    ax2 = ax.twinx()
    ax2.plot(arr_rmse_survivors_df['survivors'],c='green',label='Routes Retained')
    ax2.set_ylim(0,30000)
    ax2.set_ylabel('Routes retained')
    ax2.grid(False)

    #Combining both lines into one legend
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc='upper right')

    plt.show()

def unq_u_r(df):
    '''
    Return number of users and routes in df
    '''
    return df['user_id'].nunique(), df['route'].nunique()


##########################################################################################################

#Functions to plot a range of numbers with a step (i.e. instead of trying every combination of numbers from 1-30 (which is VERY computationally expensive) just try 1,5,10,15,20,25,30). This is good to use if you dont have enough memory

#Still need to fine tune the graphs! Something is happening with the survivorship

def thresh_metrics(df, u=5, r=5, rn1=0, rn2=0):
    '''
    Compute metrics for every combination of thresholds
    ---Parameters---
    df (Pandas DataFrame) RUS DataFrame
    u (int) Number of ratings threshold for users
    r (int) Number of ratings threshold for routeIDs
    rn1, rn2= numbers to enter for the start & end of range function in case you dont want to interate through every number
    ---Returns---
    arr_rmse, arr_mae, arr_fcp (tuple of np-arrays)
    '''
    arr_rmse = np.zeros((u,r))
    arr_mae = np.zeros((u,r))
    arr_fcp = np.zeros((u,r))
    for i in range(rn1, u, rn2):
        for j in range(rn1, r, rn2):
            df_chopped = rus_chop(df,i,j)
            rmse, mae, fcp = algo_metrics(df_chopped)
            arr_rmse[i,j] = rmse
            arr_mae[i,j] = mae
            arr_fcp[i,j] = mae
    return arr_rmse, arr_mae, arr_fcp

def plot_RMSE_route(df, arr_rmse, title, rn1=0,rn2=0, rn3=0, u=0):
    '''
    Plot RMSE by route-coldstart threshold; assumes user threshold = u

    rn1, rn2, rn3= numbers to enter for the range function in case you dont want to interate through every number

    title= enter a string for the title of your graph

    arr_rmse= rmse array from thresh_metrics_arrs function
    '''
    arr_rmse_df = pd.DataFrame(arr_rmse)

    route_survivors = []
    for r in range(rn1,rn2, rn3):
        route_survivors.append(unq_u_r(rus_chop(df,u,r))[1])

    route_survivors_df = pd.DataFrame(route_survivors,columns=['survivors'])
    arr_rmse_df= arr_rmse_df[arr_rmse_df[2] >0].reset_index()

    combined_df = pd.concat([arr_rmse_df, route_survivors_df], axis= 1, ignore_index=True)
    combined_df=  combined_df.set_index(0)


    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(combined_df[3],c='blue',label='RMSE')
    ax.set_title(title, size=16)
    ax.set_ylim(.6,.68)
    ax.set_xlabel('Cold-start threshold')
    ax.set_ylabel('RMSE')
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(combined_df[23],c='green',label='Routes Retained')
    ax2.set_ylim(0,45000)
    ax2.set_ylabel('Routes retained')
    ax2.grid(False)

def plot_RMSE_user(df, arr_rmse, title, rn1= 0,rn2=0, rn3=0, r=0):
    '''
    Plot RMSE by user-coldstart threshold; assumes route threshold = r
    rn1, rn2, rn3= numbers to enter for the range function in case you dont want to interate through every number
    title= enter a string for the title of your graph
    arr_rmse= rmse array from thresh_metrics_arrs function
    '''

    arr_rmse_df = pd.DataFrame(arr_rmse)

    usersurvivors = []
    for u in range(rn1,rn2, rn3):
        usersurvivors.append(unq_u_r(rus_chop(df,u,r))[0])

    usersurvivors_df = pd.DataFrame(usersurvivors,columns=['survivors'])
    arr_rmse_df= arr_rmse_df[arr_rmse_df[2] >0].reset_index()

    combined_df = pd.concat([arr_rmse_df, usersurvivors_df], axis= 1, ignore_index=True)
    combined_df=  combined_df.set_index(0)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111)
    ax.plot(combined_df[3],c='blue',label='RMSE')
    ax.set_title(title, size=16)
    ax.set_ylim(.6,.68)
    ax.set_xlabel('Cold-start threshold')
    ax.set_ylabel('RMSE')
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(combined_df[23],c='green',label='Users Retained')
    ax2.set_ylim(0,45000)
    ax2.set_ylabel('Users retained')
    ax2.grid(False)
