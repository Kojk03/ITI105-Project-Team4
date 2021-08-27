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

class NearestNeighbour:
    def __init__(self, df, dic2):
        self.df = df
#         self.dummy = self.createDummy(df)
        self.users = df['userId'].unique()
        self.dimensions = len(df['userId'].unique())
        self.movies = df['movieId'].unique()
        self.dic2 = dic2
        self.prepareData()
        
    def prepareData(self):
        with open("Scaled2.p", "rb") as rp:
            self.dic = pickle.load(rp)
        with open("Usermean.p", "rb") as rp:
            self.meandic = pickle.load(rp)        
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
#     def extractUIMatrix3(self,user):
#         f = h5py.File('C:/Temp/Unscaled2.h5', 'r')
#         dset = f['Unscaled2']
#         index = self.dic[user]
#         array = dset[index]
#         f.close()
#         return array

    def extractUIMatrixItem(self, movieID):
        f = h5py.File('C:/Temp/Scaled2.h5', 'r')
        dset = f['Scaled2']
        index = self.dic[movieID]
        array = dset[index]
        f.close()
        return array
    
    def generateApprox(self, item):
        v = self.extractUIMatrixItem(item)
        self.engine.store_vector(v, item)
            
    def approxSimilaritiesFit(self,hashname):
        print('No. of movies in list: ', len(self.movies))
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
            for movie in self.movies:
                self.generateApprox(movie)
            redis_storage.store_hash_configuration(lshash)

        else:
            # Config is existing, create hash with None parameters
            lshash = RandomBinaryProjections(None, None, rand_seed=42)
            # Apply configuration loaded from redis
            lshash.apply_config(config)       
            self.engine = Engine(self.dimensions, lshashes=[lshash], storage=redis_storage)
            
    def approxSimilaritiesPredict(self, movie1):
        # Get nearest neighbours
        query = self.extractUIMatrixItem(movie1)
        return self.engine.neighbours(query)

    def approxSimilaritiesCandidatesCount(self, movie1):
        query = self.extractUIMatrixItem(movie1)
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
    
    def generateRatings2(self, ann, movie1):
        array = []
#         uiMatrix = self.dummy
        for movie in ann['movie']:
            series = self.extractUIMatrixItem(movie)
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
        array = self.dropMovies2(array,movie1)
        #print(len(array))
        users = np.sort(self.users)
        #print(movies)
        df = pd.DataFrame(array, columns = users)
        df= df.replace(0,np.nan)
#         df = df.head(50)
        ratings = df.mean(axis = 0)
                
        return ratings
    
    def dropMovies2(self,array,movie):
        ref = self.extractUIMatrixItem(movie)
        ref = np.nan_to_num(ref, nan=0)
        arr = np.nan_to_num(array, nan=0)
#         delete = []
        dic = {}
        for i in range(len(arr)):
            distance = scipy.spatial.distance.cosine(arr[i],ref)
            dic[i]=distance
               
        distancedf = pd.DataFrame.from_dict(dic,orient='Index')
        distancedf['user']= distancedf.index
        distancedf['distance']=distancedf[0]
        distancedf = distancedf[distancedf['distance']<1]
        distancedf = distancedf.drop(0,axis=1)
        distancedf = distancedf.sort_values(by='distance',ascending=True)
        distancedf = distancedf.head(50)
#         output = np.delete(array,delete,0)
        j=0
        for i in distancedf['user']:
            if j ==0:
                output = array[i].reshape(1,-1)
                j=1
            else:
                output = np.append(output,array[i].reshape(1,-1),axis=0)           
        return output
    
#     def userRatingPredict(self, user, movie):
        
# #         userItems = df_test[df_test['userId']==user]
# #         movieList = userItems['movieId'].unique()

#         neighbours = self.approxSimilaritiesPredict1(user)
#         ann = self.returnTopKNeighbours(neighbours,user)
#         output = self.generateRatings(ann, movie)
        
#         return output
    
    def userRatingPredict2(self, movie):
        
#         userItems = df_test[df_test['userId']==user]
#         movieList = userItems['movieId'].unique()

        neighbours = self.approxSimilaritiesPredict(movie)
        ann = pd.DataFrame(neighbours, columns =['movie'])
        output = self.generateRatings2(ann,movie)
        
        return output
    
    def calculateScore(self, movie, df_test):
        movieItems = df_test[df_test['movieId']==movie]
        userList = movieItems['userId'].unique()
        

        ratings = self.userRatingPredict2(movie)
        total = 0
        length = len(userList)
        #print(length)
        for user in userList:
            if user in self.users:
                if np.isnan(ratings[user]):
                    #print('movie do not exist in recommendations')
                    r = 0
                else:
                    r = ratings[user]
                total = total + ((r-movieItems[movieItems['userId']==user]['rating'].values[0])**2)
                #print(r)
                #print(total)
            else:
                print('User not in original list')
                length = length -1
              
        return [total,length]
    
#     def setTest(self,df_test):
#         self.df_test=df_test
#         self.userItems = self.df_test['userId'].unique()

    def generateNeighbours(self, df_test, n):
        movielist = df_test['movieId'].unique()
        dic2 = {}
        random.seed(10)
        for _ in range(n):
            index = random.randint(0,len(movielist)+1)
            movie = movielist[index]
            if movie in self.movies:
                movieItems = df_test[df_test['movieId']==movie]
                userList = movieItems['userId'].unique()
                neighbours = self.approxSimilaritiesPredict(movie)
                dic2[movie] = neighbours
            else:
                print("no such movie in database")
        
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
    
    def calculateScore2(self, df_test,n):
        movielist = df_test['movieId'].unique()
        
#         self.dic2 ={}
        movies = []
        random.seed(20)
        for _ in range(n):
            index = random.randint(0,len(movielist)+1)
            movie = movielist[index]
            if movie in self.movies:
                movies.append(movie)
# #        for user in userlist: 
#             if user in self.users:
#                 userItems = df_test[df_test['userId']==user]
#                 movieList = userItems['movieId'].unique()
#                 neighbours = self.approxSimilaritiesPredict(user)
#                 self.dic2[user] = neighbours
#             else:
#                 print("no user in database")
        
        print(movies)
        print("Starting MP")
        p = Pool(8)
        ratings = p.map(self.generateRatingsMP, movies)
        
        grandtotalMSE = 0
        grandtotalMAE = 0
        grandtotalMSE2 = 0
        grandtotalMAE2 = 0
        grandlength = 0
        grandlength2 = 0
        grandnorec = 0
        
        print('Starting score calculations')
        
        for i in range(len(ratings)):
            movieid = ratings[i][0]
            movieItems = df_test[df_test['movieId']==movieid]
            userList = movieItems['userId'].unique()
            
            movieratings = ratings[i][1]
            
            norec = 0
            totalMSE = 0
            totalMAE = 0
            totalMSE2 = 0
            totalMAE2 = 0
            length = len(userList)
            length2 = len(userList)
            #print(length)
            
            for user in userList:
                if user in self.users:
                    ref = movieItems[movieItems['userId']==user]['rating'].values[0]
                    if np.isnan(movieratings[user]):
                        #print('movie do not exist in recommendations')
                        norec = norec +1
                        r = 0 
                        length = length -1
                    else:
                        r = movieratings[user]  + self.meandic[user]
                        totalMSE = totalMSE + ((r-ref)**2)
                        totalMAE = totalMAE + abs((r-ref))
                    totalMSE2 = totalMSE2 + ((r-ref)**2)
                    totalMAE2 = totalMAE2 + abs((r-ref))
                    #print(r)
                    #print(total)
                else:
                    print('user not in original list')
                    length = length -1
                    length2 = length2 -1
            
            grandtotalMSE = grandtotalMSE + totalMSE
            grandtotalMAE = grandtotalMAE + totalMAE
            grandlength = grandlength + length
            grandtotalMSE2 = grandtotalMSE2 + totalMSE2
            grandtotalMAE2 = grandtotalMAE2 + totalMAE2
            grandlength2 = grandlength2 + length2
            grandnorec = grandnorec + norec
        
        return grandtotalMSE, grandtotalMAE, grandlength, grandnorec, grandtotalMSE2, grandtotalMAE2, grandlength2
    
    def generateRatingsMP(self,movie):
        neighbours = self.dic2[movie]
        ann = pd.DataFrame(neighbours, columns =['movie'])
        output = self.generateRatings2(ann,movie)
        return [movie,output]