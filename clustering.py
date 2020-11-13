# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 13:47:20 2020

@author: lrzma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math


#data=pd.read_csv('bbchealth.csv')

def preprocess(data):
   del data['Time']
   del data['No']
   data['Tweet'] = data['Tweet'].str.replace('http\S+|www.\S+', '', case=False)
   data.astype(str).apply(lambda Tweet: Tweet.str.lower())
   data.head()
   return data

#k=3
def getCentroids(k):
    k=6
    np.random.seed(200)
    centroids={
        i+1:[np.random.randint(0,80),
             np.random.randint(0,80)]
    for i in range(k)
        }
    fig=plt.figure(figsize=(50,50))
    plt.scatter(data['Tweet'],data['Number'],color='k')
    colormap={1:'r',2:'g',3:'b',4:'m',5:'c',6:'y'}
    for i in centroids.keys():
        plt.scatter(*centroids[i],color=colormap[i])
    plt.xlim(0,80)
    plt.ylim(0,80)
    return plt.show()
    
"""
    N = 9
    x = np.random.rand(N)
    y = np.random.rand(N)

    plt.scatter(x,y)
    plt.show()
    
    return plt.show()
"""


def k_means(tweets, k=4, max_iterations=20):

    centroids = []
    tweets=data['Tweet']
    i = 0
    hash_map = dict()
    while i < k:
        random_tweet_id = np.random.randint(0, len(tweets) - 1)
        if random_tweet_id not in hash_map:
            i += 1
            hash_map[random_tweet_id] = True
            centroids.append(tweets[random_tweet_id])

    iteration = 0
    prev_centroids = []
    while (is_converged(prev_centroids, centroids)) == False and (iteration < max_iterations):
        print("running iteration " + str(iteration))
        clusters = assign_cluster(tweets, centroids)
        prev_centroids = centroids
        centroids = update_centroids(clusters)
        iteration = iteration + 1

    if (iteration == max_iterations):
        print("max iterations reached, K means not converged")
    else:
        print("converged")

    sse = compute_SSE(clusters)

    return clusters, sse


def is_converged(prev_centroid, new_centroids):
    
    
    if len(prev_centroid) != len(new_centroids):
        return False
    for c in range(len(new_centroids)):
        if " ".join(new_centroids[c]) != " ".join(prev_centroid[c]):
            return False
    return True


def assign_cluster(tweets, centroids):

    clusters = dict()
    for t in range(len(tweets)):
        minimum_distance = math.inf
        cluster_id = -1;
        for c in range(len(centroids)):
            distance = getDistance(centroids[c], tweets[t])

            if centroids[c] == tweets[t]:
                print("tweet and centroid are equal with c: " + str(c) + ", t" + str(t))
                cluster_id = c
                minimum_distance = 0
                break

            if distance < minimum_distance:
                cluster_id = c
                minimum_distance = distance

       
        if minimum_distance == 1:
            cluster_id = np.radom.randint(0, len(centroids) - 1)
        clusters.setdefault(cluster_id, []).append([tweets[t]])
        print("tweet t: " + str(t) + " is assigned to cluster c: " + str(cluster_id))
        last_tweet_id = len(clusters.setdefault(cluster_id, [])) - 1
        clusters.setdefault(cluster_id, [])[last_tweet_id].append(minimum_distance)
        
        

    return clusters


def update_centroids(clusters):

    centroids = []
    for c in range(len(clusters)):
        SumOfMinimumDistance = math.inf
        centroid_id = -1
        minimum_distance_dp = []
        for t1 in range(len(clusters[c])):
            minimum_distance_dp.append([])
            SumOfDistance = 0
            for t2 in range(len(clusters[c])):
                if t1 != t2:
                    if t2 < t1:
                        distance = minimum_distance_dp[t2][t1]
                    else:
                        distance = getDistance(clusters[c][t1][0], clusters[c][t2][0])

                    minimum_distance_dp[t1].append(distance)
                    SumOfDistance  += distance
                else:
                    minimum_distance_dp[t1].append(0)
        
        if SumOfDistance <  SumOfMinimumDistance:
           SumOfMinimumDistance  = SumOfDistance
           centroid_id = t1
    
        centroids.append(clusters[c][centroid_id][0])
    
    return centroids
            
           
def getDistance(tweet1, tweet2):

    
    intersection = set(tweet1).intersection(tweet2)
    union = set().union(tweet1, tweet2)
    return 1 - (len(intersection) / len(union))


def compute_SSE(clusters):

    sse = 0
    for c in range(len(clusters)):
        for t in range(len(clusters[c])):
            sse = sse + (clusters[c][t][1] * clusters[c][t][1])

    return sse


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
           
            #for t in range(len(clusters[c])):
                #print("t" + str(t) + ", " + (" ".join(clusters[c][t][0])))

        print("--> SSE : " + str(sse))
        print('\n')
        k = k + 1