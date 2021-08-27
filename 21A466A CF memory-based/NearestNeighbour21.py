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

# A seperate class is used to carried out multiprocessing. Multiprocessing doesn't work on objects connected to Redis. Could have just instantialize a new object with the same class.

class NearestNeighbour:
    def __init__(self, df,dic2):
        self.df = df
#         self.dummy = self.createDummy(df)
        self.users = df['userId'].unique()
        self.dimensions = len(df['movieId'].unique())
        self.movies = df['movieId'].unique()
        self.dic2 = dic2
        self.prepareData()
        
    def prepareData(self):
        with open("Scaled.p", "rb") as rp:
            self.dic = pickle.load(rp)
        with open("Usermean.p", "rb") as rp:
            self.meandic = pickle.load(rp)  
#         self.f = h5py.File('D:/Unscaled.h5', 'r')
#         self.dset = self.f['Unscaled']

    
    def closeFile(self):
        self.f.close()        
    
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
            lshash = RandomBinaryProjections(hashname, 8, rand_seed=42)
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
    
    def generateRatings2(self, ann, user):
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
        array = self.dropUsers2(array,user)
        #print(len(array))
        movies = np.sort(self.movies)
        #print(movies)
        df = pd.DataFrame(array, columns = movies)
        df= df.replace(0,np.nan)
        ratings = df.mean(axis = 0)
                
        return ratings

    def generateRatings3(self, ann, user):
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
    
#     def userRatingPredict(self, user, movie):
        
# #         userItems = df_test[df_test['userId']==user]
# #         movieList = userItems['movieId'].unique()

#         neighbours = self.approxSimilaritiesPredict(user)
#         ann = self.returnTopKNeighbours(neighbours,user)
#         output = self.generateRatings(ann, movie)
        
#         return output
    
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
        #print(length)
        for movie in movieList:
            if movie in self.movies:
                if np.isnan(ratings[movie]):
                    #print('movie do not exist in recommendations')
                    r = 0
                else:
                    r = ratings[movie]
                total = total + ((r-userItems[userItems['movieId']==movie]['rating'].values[0])**2)
                #print(r)
                #print(total)
            else:
                print('Movie not in original list')
                length = length -1
              
        return [total,length]
    
#     def setTest(self,df_test):
#         self.df_test=df_test
#         self.userItems = self.df_test['userId'].unique()

    def calculateScore2(self, df_test,n):
        userlist = df_test['userId'].unique()
        
#         self.dic2 ={}
        users = []
        random.seed(20)
        for _ in range(n):
            index = random.randint(0,len(userlist)+1)
            user = userlist[index]
            if user in self.users:
                users.append(user)
# #        for user in userlist: 
#             if user in self.users:
#                 userItems = df_test[df_test['userId']==user]
#                 movieList = userItems['movieId'].unique()
#                 neighbours = self.approxSimilaritiesPredict(user)
#                 self.dic2[user] = neighbours
#             else:
#                 print("no user in database")
        
        print(users)
#         print("Starting MP")
#         p = Pool(12)
#         ratings = p.map(self.generateRatingsMP, users)

        ratings = []
        
        for user in users:
            neighbours = self.dic2[user]
            ann = pd.DataFrame(neighbours, columns =['user'])
            output = self.generateRatings2(ann,user)
            ratings.append([user,output])
        
    
        grandtotalMSE = 0
        grandtotalMAE = 0
        grandtotalMSE2 = 0
        grandtotalMAE2 = 0
        grandlength = 0
        grandlength2 = 0
        grandnorec = 0
        
        
        for i in range(len(ratings)):
            userid = ratings[i][0]
            userItems = df_test[df_test['userId']==userid]
            movieList = userItems['movieId'].unique()
            
            userratings = ratings[i][1]
            
            norec = 0
            totalMSE = 0
            totalMAE = 0
            totalMSE2 = 0
            totalMAE2 = 0
            length = len(movieList)
            length2 = len(movieList)
            #print(length)
            
            for movie in movieList:
                if movie in self.movies:
                    ref = userItems[userItems['movieId']==movie]['rating'].values[0]
                    if np.isnan(userratings[movie]):
                        #print('movie do not exist in recommendations')
                        norec = norec +1
                        r = 0 
                        length = length -1
                    else:
                        r = userratings[movie]  + self.meandic[userid]
                        totalMSE = totalMSE + ((r-ref)**2)
                        totalMAE = totalMAE + abs((r-ref))
                    totalMSE2 = totalMSE2 + ((r-ref)**2)
                    totalMAE2 = totalMAE2 + abs((r-ref))
                    #print(r)
                    #print(total)
                else:
                    print('movie not in original list')
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
    
    def generateRatingsMP(self,user):
        neighbours = self.dic2[user]
        ann = pd.DataFrame(neighbours, columns =['user'])
        output = self.generateRatings2(ann,user)
        return [user,output]