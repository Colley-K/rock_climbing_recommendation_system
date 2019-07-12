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

It doesn’t rely of features of the item, but the preferences from other users. Similar users survey needs to be done.

![](./6_README_files/matrix_example.png)

## 3. Data Cleaning 

[Data Cleaning Report](https://drive.google.com/open?id=195wcooDtT2XhfpRXREWmLovm8XZPNymy)





