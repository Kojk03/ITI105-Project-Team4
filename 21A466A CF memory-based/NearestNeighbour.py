#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import scipy
import redis
import pickle
import h5py
from scipy import spatial
from multiprocessing import Pool
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter
from nearpy import storage

# For KNN and for creation of h5py files

class NearestNeighbour:
    def __init__(self, df):
        self.df = df
        self.dummy = self.createDummy(df)
        self.dummyItem = self.createDummyItem(df)
        self.users = df['userId'].unique()
        self.dimensions = len(df['movieId'].unique())
        self.movies = df['movieId'].unique()
        #self.prepareData()
        self.prepareData()
        
    def prepareData(self):
        with open("Scaled.p", "rb") as rp:
            self.dic = pickle.load(rp)        
#     def prepareData(self):
#         array = []
#         for user in self.users:
#             print(user)
#             userMatrix = self.extractUIMatrix(user).values
#             userMatrix = userMatrix.reshape(1,-1)
#             if len(array)!=0:
#                 array = np.append(array, userMatrix, axis = 0)
#             else:
#                 array = userMatrix
#             print(array)
    
    def extractUIMatrix(self, userID):
        userdf = self.df[self.df['userId']==userID]
        userMatrix = pd.pivot_table(userdf,values='rating',columns='movieId',index='userId', dropna = False)
        userMatrix = pd.concat([userMatrix,self.dummy])
        userMatrix = userMatrix.fillna(0)
        userMatrix = userMatrix.iloc[0]
        return userMatrix
    
    def extractUIMatrixItem(self, movieID):
        itemdf = self.df[self.df['movieId']==movieID]
        itemMatrix = pd.pivot_table(itemdf,values='rating',columns='userId',index='movieId', dropna = False)
        itemMatrix = pd.concat([itemMatrix,self.dummyItem])
        itemMatrix = itemMatrix.fillna(0)
        itemMatrix = itemMatrix.iloc[0]
        return itemMatrix
    
    def extractUIMatrix2(self, userID):
        userdf = self.df[self.df['userId']==userID]
        userMatrix = pd.pivot_table(userdf,values='rating',columns='movieId',index='userId', dropna = False)
        userMatrix = pd.concat([userMatrix,self.dummy])
        userMatrix = userMatrix.iloc[0]
        return userMatrix

    
    def extractUIMatrix2Scaled(self, userID):
        userdf = self.df[self.df['userId']==userID]
        userMatrix = pd.pivot_table(userdf,values='rating',columns='movieId',index='userId', dropna = False)
        userMatrix = pd.concat([userMatrix,self.dummy])
        userMatrix = userMatrix.iloc[0]
        
        useravg = userMatrix.mean()
        userMatrix = userMatrix.apply(lambda x: x - useravg)

        return userMatrix
    
    def extractUIMatrix3(self,user):
        f = h5py.File('C:/Temp/Scaled.h5', 'r')
        dset = f['Scaled']
        index = self.dic[user]
        array = dset[index]
        f.close()
        return array    
    
    def createDummy(self, df):
        movies = df['movieId'].unique()
        dummy = pd.DataFrame(movies,columns=['movieId'])
        dummy['userId']=0
        dummy['rating']=1
        dummy = dummy.pivot_table(values='rating',index='userId',columns='movieId')
        return dummy
    
    def createDummyItem(self, df):
        users = df['userId'].unique()
        dummy = pd.DataFrame(users,columns=['userId'])
        dummy['movieId']=0
        dummy['rating']=1
        dummy = dummy.pivot_table(values='rating',index='movieId',columns='userId')
        return dummy

    def cosineSimilarity(self, user1, user2):
        array1 = self.extractUIMatrix(user1).values
        array2 = self.extractUIMatrix(user2).values
        similarity = 1 - scipy.spatial.distance.cosine(array1,array2)
        return similarity
    
    def cosineSimilarity2(self, user1, user2):
        array1 = self.extractUIMatrix3(user1)
        array1 = np.nan_to_num(array1, nan=0)
        array2 = self.extractUIMatrix3(user2)
        array2 = np.nan_to_num(array2, nan=0)
        similarity = 1 - scipy.spatial.distance.cosine(array1,array2)
        return similarity
    
    def generate(self, user):
#         print(user)
        distance = 1
        if self.user1 != user:
            distance = self.cosineSimilarity2(self.user1, user)
#             print(distance)
#             for i in range(self.toplist):
#                 if distance > self.nearestlist[i][1]:
#                     if i == self.toplist-1:
#                         self.nearestlist[i] = [user,distance]
#                     else:
#                         j = self.toplist-1
#                         while j > i:
#                             self.nearestlist[j] = self.nearestlist[j-1]
#                             j = j -1
#                         self.nearestlist[i] = [user,distance]
        return [user,distance]
    
    def distancelist(self):
        p = Pool(12)
        return p.map(self.generate, self.users)        
    
    def nearestSimilarities(self, toplist, user1):
        print('No. of users in list: ', len(self.users))
        if user1 in self.users:
            print('Not new user')
            self.toplist = toplist
            self.user1 = user1            
#             self.nearestlist = []
#             for i in range(self.toplist):
#                 self.nearestlist.append([0,0])  
            distancelist = self.distancelist()
#             rdf = pd.DataFrame(distancelist)
#             rdf = rdf.sort_values(1,ascending=False)
#             return rdf.iloc[:self.toplist,:]
            return distancelist
        else:
            print('Is new user')
            
    def nearestSimilaritiesNoMP(self, toplist, user1):
        print('No. of users in list: ', len(self.users))
        if user1 in self.users:
            print('Not new user')
            self.toplist = toplist
            self.user1 = user1
            self.nearestlist = []
            
            flag=0
            for user in self.users:
                distance = 1
                if self.user1 != user:
                    distance = self.cosineSimilarity(self.user1, user)
                    if flag==0:
                        nearestlist=[user,distance]
                    else:
                        self.nearestlist.append([user,distance])
                    flag=1
            rdf = pd.DataFrame(self.nearestlist)
            rdf = rdf.sort_values(1,ascending=False)
            return rdf.iloc[:self.toplist,:]
        else:
            print('Is new user')
    
    def generateApprox(self, user):
        v = self.extractUIMatrix(user)
        self.engine.store_vector(v.to_numpy(), user)
            
    def approxSimilaritiesFit(self,hashname):
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
            lshash = RandomBinaryProjections(hashname, 10, rand_seed=42)
            self.engine = Engine(self.dimensions, lshashes=[lshash], storage=redis_storage)
            for user in self.users:
                self.generateApprox(user)
            redis_storage.store_hash_configuration(lshash)

        else:
            # Config is existing, create hash with None parameters
            lshash = RandomBinaryProjections(None, None, rand_seed=42)
            # Apply configuration loaded from redis
            lshash.apply_config(config)       
            self.engine = Engine(self.dimensions, lshashes=[lshash], storage=redis_storage)
            
    def approxSimilaritiesPredict(self, user1):
        # Get nearest neighbours
        query = self.extractUIMatrix(user1)
        return self.engine.neighbours(query.to_numpy())

    def approxSimilaritiesCandidatesCount(self, user1):
        query = self.extractUIMatrix(user1)        
        return self.engine.candidate_count(query.to_numpy())
    
#     def returnTopKNeighbours(self, neighbours, user1):
#         neighbours = pd.DataFrame(neighbours, columns =['user'])
#         #neighbours['distance']=neighbours['user'].apply(self.cosineSimilarity, args=(user1,))      
#         return neighbours

    def dropUsers(self, neighbours,item):
        for user in neighbours['user']:
            uiMatrix = self.extractUIMatrix(user)
            #print(uiMatrix[item])
            if uiMatrix[item] == 0:
                #print('dropping', user)
                neighbours = neighbours.drop(neighbours[neighbours['user']==user].index)
        return neighbours

    def generateRatings(self, ann, movie):
        ratings = []
        l = self.dropUsers(ann, movie)
        for user in l['user']:
            uiMatrix = self.extractUIMatrix(user)
            ratings.append(uiMatrix[movie])
        return [movie,np.mean(ratings)]
    
    def generateRatings2(self, ann, user):
        array = []
        uiMatrix = self.dummy
        for user in ann:
            series = self.extractUIMatrix3(user)
#                 uiMatrix = pd.concat([series,self.dummy])
#                 uiMatrix = uiMatrix.drop([0])
#                 uiMatrix = pd.concat([uiMatrix, series])
#             uiMatrix = series.values
            uiMatrix = series.reshape(1,-1)
            if len(array)!=0:
                array = np.append(array, uiMatrix, axis = 0)
            else:
                array = uiMatrix
        #array = self.dropUsers2(array,user)
        #print(len(array))
        movies = np.sort(self.movies)
        df = pd.DataFrame(array, columns = movies)
        df= df.replace(0,np.nan)
        ratings = df.mean(axis = 0)
                
        return ratings

    
    def dropUsers2(self,array,user):
        ref = self.extractUIMatrix2(user).values.reshape(1,-1)[0]
        ref = np.nan_to_num(ref, nan=0)
        arr = np.nan_to_num(array, nan=0)
        delete = []
        for i in range(len(arr)):
            distance = scipy.spatial.distance.cosine(arr[i],ref)
            if distance ==1:
                delete.append(i)
#             if distance ==0:
#                 delete.append(i)
        output = np.delete(array,delete,0)
        return output
    
    def userRatingPredict(self, user, movie):
        
#         userItems = df_test[df_test['userId']==user]
#         movieList = userItems['movieId'].unique()

        neighbours = self.approxSimilaritiesPredict(user)
        ann = self.returnTopKNeighbours(neighbours,user)
        output = self.generateRatings(ann, movie)
        
        return output
    
    def userRatingPredict2(self, user):
        
#         userItems = df_test[df_test['userId']==user]
#         movieList = userItems['movieId'].unique()

        neighbours = self.approxSimilaritiesPredict(user)
        ann = pd.DataFrame(neighbours, columns =['user'])
        output = self.generateRatings2(ann,user)
        
        return output
    
    def calculateScore(self, user, df_test):
        userItems = df_test[df_test['userId']==user]
        movieList = userItems['movieId'].unique()
        

        ratings = self.userRatingPredict2(user)
        total = 0
        length = len(movieList)
        for movie in movieList:
            if movie in self.movies:
                if np.isnan(ratings[movie]):
                    #print('movie do not exist in recommendations')
                    r = 0
                else:
                    r = ratings[movie]
                total = total + abs((r-userItems[userItems['movieId']==movie]['rating'].values[0]))
                #print(r)
                #print(total)
            else:
                print('Movie not in original list')
                length = length -1
              
        return total/length
    
    def setTest(self,df_test):
        self.df_test=df_test
        self.userItems = self.df_test['userId'].unique()