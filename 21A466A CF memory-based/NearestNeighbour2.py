#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import scipy
import redis
from scipy import spatial
from multiprocessing import Pool
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter
from nearpy import storage
import pickle
import h5py
import random

#For user-based 

class NearestNeighbour:
    def __init__(self, df):
        self.df = df
#         self.dummy = self.createDummy(df)
        self.users = df['userId'].unique()
        self.dimensions = len(df['movieId'].unique())
        self.movies = df['movieId'].unique()
        self.prepareData()
        
    def prepareData(self):
        with open("Scaled.p", "rb") as rp:
            self.dic = pickle.load(rp)
#         self.f = h5py.File('D:/Unscaled.h5', 'r')
#         self.dset = self.f['Unscaled']
    
#     def closeFile(self):
#         self.f.close()        
    
#     def extractUIMatrix(self, userID):
#         userdf = self.df[self.df['userId']==userID]
#         userMatrix = pd.pivot_table(userdf,values='rating',columns='movieId',index='userId', dropna = False)
#         userMatrix = pd.concat([userMatrix,self.dummy])
#         userMatrix = userMatrix.fillna(0)
#         userMatrix = userMatrix.iloc[0]
#         return userMatrix
    
#     def extractUIMatrix2(self, userID):
#         userdf = self.df[self.df['userId']==userID]
#         userMatrix = pd.pivot_table(userdf,values='rating',columns='movieId',index='userId', dropna = False)
#         userMatrix = pd.concat([userMatrix,self.dummy])
#         userMatrix = userMatrix.iloc[0]
#         return userMatrix
    
#     def createDummy(self, df):
#         movies = df['movieId'].unique()
#         dummy = pd.DataFrame(movies,columns=['movieId'])
#         dummy['userId']=0
#         dummy['rating']=1
#         dummy = dummy.pivot_table(values='rating',index='userId',columns='movieId')
#         return dummy

#     def cosineSimilarity(self, user1, user2):
#         array1 = self.extractUIMatrix(user1).values
#         array2 = self.extractUIMatrix(user2).values
#         similarity = 1 - scipy.spatial.distance.cosine(array1,array2)
#         return similarity
    
#     def generate(self, user):
# #         print(user)
#         distance = 1
#         if self.user1 != user:
#             distance = self.cosineSimilarity(self.user1, user)
# #             print(distance)
# #             for i in range(self.toplist):
# #                 if distance > self.nearestlist[i][1]:
# #                     if i == self.toplist-1:
# #                         self.nearestlist[i] = [user,distance]
# #                     else:
# #                         j = self.toplist-1
# #                         while j > i:
# #                             self.nearestlist[j] = self.nearestlist[j-1]
# #                             j = j -1
# #                         self.nearestlist[i] = [user,distance]
#         return [user,distance]
    
#     def distancelist(self):
#         p = Pool(12)
#         return p.map(self.generate, self.users)        
    
#     def nearestSimilarities(self, toplist, user1):
#         print('No. of users in list: ', len(self.users))
#         if user1 in self.users:
#             print('Not new user')
#             self.toplist = toplist
#             self.user1 = user1            
# #             self.nearestlist = []
# #             for i in range(self.toplist):
# #                 self.nearestlist.append([0,0])  
#             distancelist = self.distancelist()
# #             rdf = pd.DataFrame(distancelist)
# #             rdf = rdf.sort_values(1,ascending=False)
# #             return rdf.iloc[:self.toplist,:]
#             return distancelist
#         else:
#             print('Is new user')
            
#     def nearestSimilaritiesNoMP(self, toplist, user1):
#         print('No. of users in list: ', len(self.users))
#         if user1 in self.users:
#             print('Not new user')
#             self.toplist = toplist
#             self.user1 = user1
#             self.nearestlist = []
            
#             flag=0
#             for user in self.users:
#                 distance = 1
#                 if self.user1 != user:
#                     distance = self.cosineSimilarity(self.user1, user)
#                     if flag==0:
#                         nearestlist=[user,distance]
#                     else:
#                         self.nearestlist.append([user,distance])
#                     flag=1
#             rdf = pd.DataFrame(self.nearestlist)
#             rdf = rdf.sort_values(1,ascending=False)
#             return rdf.iloc[:self.toplist,:]
#         else:
#             print('Is new user')
    def extractUIMatrix3(self,user):
        f = h5py.File('C:/Temp/Scaled.h5', 'r')
        dset = f['Scaled']
        index = self.dic[user]
        array = dset[index]
        f.close()
        return array
    
    def generateApprox(self, user):
        v = self.extractUIMatrix3(user)
        self.engine.store_vector(v, user)
            
    def approxSimilaritiesFit(self,hashname,randomSeed):
        print('No. of users in list: ', len(self.users))
        # Create a random binary hash with 10 bits
#         rbp = RandomBinaryProjections('rbp', 10, rand_seed=42)
#         nf = NearestFilter(topfilter)

        redis_object = redis.StrictRedis(host='localhost', port=6379, db=0)
        redis_storage = storage.RedisStorage(redis_object)
        config = redis_storage.load_hash_configuration(hashname)
        
        # Create engine with pipeline configuration
        if config is None:
            # Config is not existing, create hash from scratch, with 10 projections
            lshash = RandomBinaryProjections(hashname, 10, rand_seed=randomSeed)
            self.engine = Engine(self.dimensions, lshashes=[lshash], storage=redis_storage)
            for user in self.users:
                self.generateApprox(user)
            redis_storage.store_hash_configuration(lshash)

        else:
            # Config is existing, create hash with None parameters
            lshash = RandomBinaryProjections(None, None, rand_seed=randomSeed)
            # Apply configuration loaded from redis
            lshash.apply_config(config)       
            self.engine = Engine(self.dimensions, lshashes=[lshash], storage=redis_storage)
            
    def approxSimilaritiesPredict(self, user1):
        # Get nearest neighbours
        query = self.extractUIMatrix3(user1)
        return self.engine.neighbours(query)

    def approxSimilaritiesCandidatesCount(self, user1):
        query = self.extractUIMatrix3(user1)
        #print(query.shape)
        return self.engine.candidate_count(query)
    
#     def returnTopKNeighbours(self, neighbours, user1):
#         neighbours = pd.DataFrame(neighbours, columns =['user'])
#         #neighbours['distance']=neighbours['user'].apply(self.cosineSimilarity, args=(user1,))      
#         return neighbours

#     def dropUsers(self, neighbours,item):
#         for user in neighbours['user']:
#             uiMatrix = self.extractUIMatrix(user)
#             #print(uiMatrix[item])
#             if uiMatrix[item] == 0:
#                 #print('dropping', user)
#                 neighbours = neighbours.drop(neighbours[neighbours['user']==user].index)
#         return neighbours

#     def generateRatings(self, ann, movie):
#         ratings = []
#         l = self.dropUsers(ann, movie)
#         for user in l['user']:
#             uiMatrix = self.extractUIMatrix(user)
#             ratings.append(uiMatrix[movie])
#         return [movie,np.mean(ratings)]
    
    def generateRatings2(self, ann, user1, k):
        array = []
#         uiMatrix = self.dummy
        for user in ann['user']:
            series = self.extractUIMatrix3(user)
            series = series.reshape(1,-1)
#                 uiMatrix = pd.concat([series,self.dummy])
#                 uiMatrix = uiMatrix.drop([0])
#                 uiMatrix = pd.concat([uiMatrix, series])
#             uiMatrix = series.values
#             uiMatrix = uiMatrix.reshape(1,-1)
            if len(array)!=0:
                array = np.append(array, series, axis = 0)
            else:
                array = series
            #print(array)
        array, distancelist = self.dropUsers2(array,user1)
        #print(len(array))
        movies = np.sort(self.movies)
        #print(movies)
        df = pd.DataFrame(array, columns = movies)
        df= df.replace(0,np.nan)
        df['distance']=distancelist
        df = df.sort_values(by ='distance')
        df = df.head(k)
        ratings = df.mean(axis = 0)
                
        return ratings
    
    
#     def dropUsers2(self,array,user):
#         ref = self.extractUIMatrix3(user)
#         ref = np.nan_to_num(ref, nan=0)
#         arr = np.nan_to_num(array, nan=0)
#         delete = []
#         distancelist = []
#         for i in range(len(arr)):
#             distance = scipy.spatial.distance.cosine(arr[i],ref)
#             if distance ==1:
#                 delete.append(i)
#             elif distance ==0:
#                 delete.append(i)
#             else:
#                 distancelist.append(distance)
#         output = np.delete(array,delete,0)
#         print('No. of neighbours: ',len(output))
#         return output, distancelist

    def dropUsers2(self,array,user):
        ref = self.extractUIMatrix3(user)
        ref = np.nan_to_num(ref, nan=0)
        arr = np.nan_to_num(array, nan=0)
#         delete = []
        dic = {}
        for i in range(len(arr)):
            distance = scipy.spatial.distance.cosine(arr[i],ref)
            dic[i]=distance
               
        distancedf = pd.DataFrame.from_dict(dic,orient='Index')
        distancedf['user']= distancedf.index
        print(distancedf)
        distancedf['distance']=distancedf[0]
        distancedf = distancedf[distancedf['distance']<1]
        distancedf = distancedf.drop(0,axis=1)
        distancedf = distancedf.sort_values(by='distance',ascending=True)
        distancedf = distancedf.head(100)
#         output = np.delete(array,delete,0)
        j=0
        output =[]
        for i in distancedf['user']:
            if j ==0:
                output = array[i].reshape(1,-1)
                j=1
            else:
                output = np.append(output,array[i].reshape(1,-1),axis=0)           
        return output
    
    
    def calculateDistances(self, neighbours, user1):
        ref = self.extractUIMatrix3(user1)
        ref = np.nan_to_num(ref, nan=0)
        
        output=[]
        
        for user in neighbours:
            arr = self.extractUIMatrix3(user)
            distance = 1 - scipy.spatial.distance.cosine(arr,ref)
            output.append([user,distance])
        
        return output
#     def userRatingPredict(self, user, movie):
        
# #         userItems = df_test[df_test['userId']==user]
# #         movieList = userItems['movieId'].unique()

#         neighbours = self.approxSimilaritiesPredict(user)
#         ann = self.returnTopKNeighbours(neighbours,user)
#         output = self.generateRatings(ann, movie)
        
#         return output
    
    def userRatingPredict2(self, user, k):
        
#         userItems = df_test[df_test['userId']==user]
#         movieList = userItems['movieId'].unique()

        neighbours = self.approxSimilaritiesPredict(user)
        ann = pd.DataFrame(neighbours, columns =['user'])
        output = self.generateRatings2(ann,user, k)
        
        return output
    
    def calculateScore(self, user, df_test, k):
        userItems = df_test[df_test['userId']==user]
        movieList = userItems['movieId'].unique()
        

        ratings = self.userRatingPredict2(user, k)
        totalMSE = 0
        totalMAE = 0
        length = len(movieList)
        #print(length)
        for movie in movieList:
            if movie in self.movies:
                if np.isnan(ratings[movie]):
                    #print('movie do not exist in recommendations')
                    r = 0
                else:
                    r = ratings[movie]
                totalMSE = totalMSE + ((r-userItems[userItems['movieId']==movie]['rating'].values[0])**2)
                totalMAE = totalMAE + abs((r-userItems[userItems['movieId']==movie]['rating'].values[0]))
                #print(r)
                #print(total)
            else:
                print('Movie not in original list')
                length = length -1
              
        return [totalMSE, totalMAE, length]
    
#     def setTest(self,df_test):
#         self.df_test=df_test
#         self.userItems = self.df_test['userId'].unique()
    def generateRatings3(self, ann, user):
        array = []
#         uiMatrix = self.dummy
        for user in ann:
            series = self.extractUIMatrix3(user)
            series = series.reshape(1,-1)
#                 uiMatrix = pd.concat([series,self.dummy])
#                 uiMatrix = uiMatrix.drop([0])
#                 uiMatrix = pd.concat([uiMatrix, series])
#             uiMatrix = series.values
#             uiMatrix = uiMatrix.reshape(1,-1)
            if len(array)!=0:
                array = np.append(array, series, axis = 0)
            else:
                array = series
            #print(array)
        array = self.dropUsers2(array,user)
        #print(len(array))
        movies = np.sort(self.movies)
        #print(movies)
        df = pd.DataFrame(array, columns = movies)
        df= df.replace(0,np.nan)
#         ratings = df.mean(axis = 0)
                
        return df
    
    def recommendations(self, neighbours, user):
        ratingsdf = self.generateRatings3(neighbours,user)
        ratingscount = ratingsdf.count()
        ratingscount = ratingscount[ratingscount >1]
        ratingsdf2 = ratingsdf[ratingscount.index]
        recommendations = ratingsdf2.mean(axis =0).sort_values(ascending=False).head(10)
        return recommendations
        

    def generateNeighbours(self, df_test, n):
        userlist = df_test['userId'].unique()
        dic2 = {}
        random.seed(20)
        for _ in range(n):
            index = random.randint(0,len(userlist)+1)
            user = userlist[index]
            if user in self.users:
                userItems = df_test[df_test['userId']==user]
                movieList = userItems['movieId'].unique()
                neighbours = self.approxSimilaritiesPredict(user)
                dic2[user] = neighbours
            else:
                print("no user in database")
        
#         ratings = self.userRatingPredict2(user)
#         total = 0
#         length = len(movieList)
#         #print(length)
#         for movie in movieList:
#             if movie in self.movies:
#                 if np.isnan(ratings[movie]):
#                     #print('movie do not exist in recommendations')
#                     r = 0
#                 else:
#                     r = ratings[movie]
#                 total = total + ((r-userItems[userItems['movieId']==movie]['rating'].values[0])**2)
#                 #print(r)
#                 #print(total)
#             else:
#                 print('Movie not in original list')
#                 length = length -1
              
        return dic2
    
#     def generateRatingsMP(self,user):
#         neighbours = self.dic2[user]
#         ann = pd.DataFrame(neighbours, columns =['user'])
#         output = self.generateRatings2(ann,user)
#         return [user,output]