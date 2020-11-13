# Clustering 
Clustering is a technique to find similarity in the dataset to classify data. A cluster can have ambiguous notation which means a group of samples can have various kinds of classification
## Clustering on pyhton
### Step 1: Preprocessing data
Delete lables of each row in the dataset. Remove timestamp and URL of the tweets.
```
   del data['Time']
   del data['No']
   data['Tweet'] = data['Tweet'].str.replace('http\S+|www.\S+', '', case=False)
   
```
### Step 2: Set up centroids
Create a method that sets up and update centroids.Meanwhile write functions that can minimize the intra-cluster distances and maximize the inter-cluster distances. In this project, we 
use Jaccard Distance metric
```
 def update_centroids(clusters):
 def getDistance(tweet1, tweet2):
 
```
### Step3: Compute SSE
```
def compute_SSE(clusters):

    sse = 0
    for c in range(len(clusters)):
        for t in range(len(clusters[c])):
            sse = sse + (clusters[c][t][1] * clusters[c][t][1])

    return sse
 ```
 ### Step4: Recall the clustering functions by setting up different k values
 In the main function, recall data preprocessing function and print out converage results. Then implement the whole projects.
 ```
 if __name__ == '__main__':

    data=pd.read_csv('bbchealth.csv')
    preprocess(data)
    tweets=data['Tweet']
    experiments = 4
    k = 7
   
    for e in range(experiments):

        print("Running K means for experiment no. " + str((e + 1)) + " for k = " + str(k))
        clusters, sse = k_means(tweets, k)
        for c in range(len(clusters)):
            print(str(c+1) + ": ", str(len(clusters[c])) + " tweets")
 ```
 ### Step5: Check and analyze the results
 ```
tweet and centroid are equal with c: 5, t3928
tweet t: 3928 is assigned to cluster c: 5
converged
1:  595 tweets
2:  360 tweets
3:  223 tweets
4:  37 tweets
5:  440 tweets
6:  451 tweets
7:  26 tweets
8:  285 tweets
9:  966 tweets
10:  546 tweets
--> SSE : 604.843896450329
```
 
 
 
 
