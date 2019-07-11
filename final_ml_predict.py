#Notebook to run the final ML recommendation and to save it for later use as well as run the predictions function

#imports
import function as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from collections import defaultdict
import surprise
from surprise import AlgoBase, Dataset, evaluate, accuracy, SVDpp, Reader, dump
from surprise.model_selection import cross_validate, KFold, train_test_split, GridSearchCV
from surprise import AlgoBase, evaluate, accuracy, SVDpp, Reader, dump
from statistics import mean, mode
import pickle

#Final ML function

df_loc = 'drive/My Drive/Capstone/data/rock6x.csv'
pickle_loc = 'drive/My Drive/Capstone/data/svdpp_algo.pkl'
algorithm = SVDpp()

def recommendation(df_loc, algorithm,  pickle_loc):
    '''
    inputs:
    alogorithm (surprise library alogorithm) whichever algo method you want to use for the recommendation system
    df_loc (string) pathway to the cleaned dataset to be used
    pickle_loc(string) pathway of where to save the pickle

    outputs:
    trained ML algo saved in the pickle_loc
    '''
    print('Reading in df and creating data object...')
    rock6x = pd.read_csv(df_loc, low_memory=False)
    data6x= f.read_data_surprise(rock6x)

    print('Train on full dataset...')
    trainset = data6x.build_full_trainset()
    algo = algorithm
    algo.fit(trainset)

    print('Pickling algorithm...')
    #pickle.dump(algo, open(pickle_loc, 'wb'))
    #print ("DONE.")

    #print (f"DONE, {algorithm} algorithm is trained and is located at: {pickle_loc}")

##########################################################################
#Top ten recommendation portion

#Creating a top_ten dataframe (HEAVY be careful what size DF you use)
def top_ten_df (df):
    '''
    inputs:
    df (Pandas DF) the dataframe that you would like to train on/NOTE: use f.df_samp_unique_vals() to get a smaller DF if you dont have enough memory to run full DF

    outputs:
    top_ten_df (DataFrame Pandas) returns a dataframe with the top ten predictions for every user in your original dataframe
    '''

    data= f.read_data_surprise(df)#use f.df_samp_unique_vals() to get a smaller DF if you dont have enough memory to run full DF

    # First train an SVD algorithm on entire dataset (choose 6x name filter)
    trainset = data.build_full_trainset()
    algo = SVDpp()#n_epochs= 18, lr_all= 0.01, reg_all= 0.175
    algo.fit(trainset)

    # Than predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()#HEAVY THIS TAKES THE MOST RAM
    predictions = algo.test(testset)

    #create a dictionary of predictions
    top_n = f.get_top_n(predictions, n=10)

    #Turn the dictionary into a df
    top_ten_df = pd.DataFrame(top_n)

    return top_ten_df



#Prettify the ref_df
def pretty_ref_df(ref_df):
    '''
    inputs:
    ref_df (Pandas DF) the original dataframe used because you trained your model

    outputs:
    ref_df (Pandas DF) returns the same dataframe, but with a better format for the final top_ten prediction function
    '''

    ref_df["route"] = ref_df["route"].str.capitalize()
    ref_df["crag"] = ref_df["crag"].str.capitalize()
    ref_df= ref_df.round({'avg_rating': 3})
    return ref_df

#Prettify the top_ten_rec_df
def pretty_top_ten_df(top_ten):
    '''
    inputs:
    top_ten (Pandas DF) the dataframe that was created from the top_ten_df function above

    outputs:
    top_ten (Pandas DF) returns the same dataframe, but with a better format for the final top_ten prediction function to work
    '''
    top_ten.iloc[:, :] = top_ten.iloc[:, :].applymap(lambda x: x.replace(",", ""))
    top_ten.iloc[:, :]= top_ten.iloc[:, :].applymap(lambda x: x.strip("()"))
    top_ten.iloc[:, :] = top_ten.iloc[:, :].applymap(lambda x: x.replace("'", ""))
    top_ten.iloc[:, :] = top_ten.iloc[:, :].applymap(lambda x: x.split(" "))
    return top_ten

#Final Prediction function
def top_ten_prediction(user_id, top_ten_df, ref_df):
    '''
    inputs:

    -user_id (string) in quotations enter the numeric user_id number to predict their top ten rock routes they would enjoy (this can be modified later to enter an actual user name that corresponds to the 8a.nu website)
    -top_ten_df (Pandas DF) created from the top_ten_df function above
    -ref_df (Pandas DF) the original dataframe that was trained on

    outputs:

    -ref_df2 (DataFrame Pandas) returns a data frame with the top ten predictions for that specific user, or if there is not enough information for that user it will return a list of the average top ten rated routes in the world

    '''
    if user_id in top_ten_df:

        #get the top ten for this user
        users_lst = top_ten_df[[user_id]]
        #create empty list
        refs = []

        #iterate through all top ten predictions
        for i in range (10):
            #look up top ten in the original DF
            row = ref_df.loc[ref_df['route'].str.lower() == (users_lst.iloc[i,0][0])].head(1)
            row['user_rate_predict'] = (users_lst.iloc[i, 0][1])
            refs.append(row)

        #Return a data frame with all the information about the climb
        ref_df2 = pd.concat(refs, axis=0, ignore_index=True).reset_index(drop=True)
        ref_df2= ref_df2[["route","user_rate_predict", "avg_rating",	"num_users_rate", "climb_type",	"usa_routes",	"usa_boulders", "crag", "country"]]
        ref_df2.index = [1,2,3,4,5,6,7,8,9,10]
        print ("Where in the world should you climb next?")
        print (f"Top Ten Rock Climbing Recommendations for User #{user_id}:")
        return ref_df2

    else:
        #Create a average top ten list dataframe sorted by any route with more the 2.9 stars and 50+ reviews
        best_ratings_df = ref_df[(ref_df["avg_rating"] > 2.9) & (ref_df["num_users_rate"] > 49)]

        #Take a random sample (so it will provide unique rec's) of ten highly rated routes
        best_ratings_df= best_ratings_df.sample(10)
        #clean up output
        best_ratings_df= best_ratings_df[["route", "avg_rating",	"num_users_rate", "climb_type",	"usa_routes",	"usa_boulders", "crag", "country"]]
        best_ratings_df.index = [1,2,3,4,5,6,7,8,9,10]
        print ("Where in the world should you climb next?")
        print (f"ERROR: User #{user_id} does not have enough historic recommendations to provide an accurate prediction")
        print (f"Average Top Ten Rock Climbing Recommendations in the World:")
        return best_ratings_df


# #create a mean column of all the rows means in the top_ten_dataframe
# from statistics import mean

# def mean_col(df):
#   emptylst= []
#   meanlst= []
#   for i in range (10):
#     for j in range(df.shape[1]-1):
#       emptylst.append(float(df.iloc[i,j][1]))
#     m = mean(emptylst)
#     m= round(m, 2)
#     meanlst.append(m)
#   print(meanlst)
#   df["mean"]= meanlst

if __name__ == '__main__':
    main()
