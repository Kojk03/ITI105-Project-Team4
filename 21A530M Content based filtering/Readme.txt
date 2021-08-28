Work contributed by 21A530M


For the experiment log, they key things tried implemented in a quick and dirty manner:

1. Alternative method of implementing content based filtering:
   - Obtain the "user feature preference matrix" for each user by multiplying the "train movie feature matrix" with the "train ratings matrix" for each user.
   - Obtaining the "closeness of the user preferences with the ratings" by multiplying the "user feature preference matrix" with the "test movie feature matrix".
   - Do a linear regression to obtain predicted ratings on the test set. 
   - Discussed with Mr Law on this implementation. Decision to stick with the K-NN implemtation instead. 
   
2. From the global movie dataset, split movies into "popular" and "unpopular" binary classes.
   Using the features, do binary classification. 
   The quick and dirty implementation did not do better than a random guess. 
   Due to time limitations, decided to focus on the recommender systems instead of the classification spin-off, which would have been interesting to see. 
