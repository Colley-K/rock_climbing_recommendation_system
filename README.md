![cover_photo](./6_README_files/cover_photo.png)
# International Rock Climbing Recommendation System

*The sport of rock climbing has been steadily increasing in popularity. From 2012-2017, the IBISWorld estimates that from average annual growth for the indoor climbing wall industry was [3.9% in the USA](https://www.ibisworld.com/industry-trends/specialized-market-research-reports/consumer-goods-services/sports-recreation/indoor-climbing-walls.html).  In 2015, it ranked 17th out of 111 out of the most popular sports in the United States ( Physical Activity Council and PHIT America). Yet, even with this growth in popularity, most of the international rock climbing websites still lack a rock climbing recommendation system. In this project, I will create a recommendation system for the 8a.nu website that will help climbers identify some unique international climbing objectives.*

## 1. Data

8a.nu is one the world’s largest international rock climbing websites. With over 4 million entries of climbs and ratings, this Kaggle webscraping project is a sufficient size to develop a good predictor model. To view the 8a.nu website, the original Kaggle four SQLite tables created by David Cohen, or the import report using the Kaggle API click on the links below:

> * [8a.nu website](https://www.8a.nu/)

> * [Kaggle Dataset](https://www.kaggle.com/dcohen21/8anu-climbing-logbook)

> * [Data Import Report](https://drive.google.com/open?id=1S4io5Nvz0lcnri_Lz9Mpa_TwLNeoSzGb)

## 2. Method

There are three main types of recommenders used in practice today:

1. **Content-based filter:** Recommending future items to the user that have similar innate features with previously "liked" items. Basically, content-based relies on similarities between features of the items & needs good item profiles to function properly.

2. **Collaborative-based filter:** Recommending products based on a similar user that has already rated the product. Collaborative filtering relies on information from similar users, and it is important to have a large explicit user rating  base (doesn't work well for new customer bases).

3. **Hybrid Method:** Leverages both content & collaborative basded filtering. Typically when a new user comes into the recommender, the content-bsaed recommendation takes place. Then after interacting with the items a couple of times, the collaborative/ user based recommendation system will be utilized.

**User-based collaborative filtering system**. 

![](./6_README_files/matrix_example.png)

I choose to work with a User-based collaborative filtering system. This made the most sense because half of the 4 million user-entered climbs had an explicit rating of how many stars the user would rate the climb. Unfortuntely, the data did not have very detailed "item features". Every rock climbing route had an area, a difficulty grade, and a style of climbing (roped or none). This would not have been enough data to provide an accurate content-based recommendation. In the future, I would love to experiment using a hybrid system to help solve the problem of the cold-start-threshold.

## 3. Data Cleaning 

[Data Cleaning Report](https://drive.google.com/open?id=195wcooDtT2XhfpRXREWmLovm8XZPNymy)

In a collaborative-filtering system there are only three columns that matter to apply the machine learning algorithms: the user, the item, and the explicit rating (see the example matrix above). I also had to clean & normalize all the reference information (location, difficulty grade, etc) to the route so that my user could get a userful and informative recommendation.

* **Problem 1:** This dataset is all user-entered information. There are a couple drop down options, but for the most part the user is able to completely make-up, or list something incorrectly. **Solution:** after normalizing & cleaning all the columns, I created a three-tier groupby system that I could then take the mode of each entry and fill in the column with that mode. For example: a route listed 12 times had the country Greece associated with it 11 times, but one person listed it in the USA. By imputing multiple columns with the mode (after the three tiered groupby), I was able to increase the accuracy of my dataset

* **Problem 2:** Being this is an international rock climbing website, the names of the rock climbing routes were differing based on if the user enters accent marks or not. **Solution:** normalize all names to the ascii standards. ascent[["name", "crag"]] = ascent[["name", "crag"]].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

* **Problem 3:** Spelling issues with the route name. For example: if there was a route named "red rocks canyon" it could be spelled "red rock", "red rocks", "red canyon" etc. **Solution:** at first I was hopeful and tried two different phonetic spelling algorithms (soundex & double metahpone). However, both of these proved to be too aggressive in their grouping and would group together sometimes 20 of the different routes as the same thing! My final solution was creating an accurate filter for route names. The logic being that if up to x users all entered that exact same route name the chances were good that it was an actual route spelled correctly. I played around with 4 different filters and kept these until I could test their prediction accruacy in the ML portion, and found the greatest prediction accuracy came from the dataset that filtered out any routes occuring less than 6 times.

## 4. EDA

Star Distributions:
![](./6_README_files/star_count.png)






