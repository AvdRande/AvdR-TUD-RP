Notes about current version of HMC-LMLP:
When using 100 samples, unlimited features, max_iter of 1000, there are many 
predictions which are good ish. Certainly at least 30% where some are entirely correct

When using 10000 samples, 1500 max features, max iter of 100, it often doesn't 
predict anything. When it does, it also isn't always correct. Also, the longest 
training time is for the first layer, as there are many features. I believe
the max amount of features is around 15000.