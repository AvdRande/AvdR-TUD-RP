Notes about current version of HMC-LMLP:
If I read the paper correctly, especially section 3.2, the predictions are not rounded
for training, but they are for the final predictions. Does this make sense?

When using 100 samples, unlimited features, max_iter of 1000, there are many 
predictions which are good ish. Certainly at least 30% where some are entirely correct

When using 10000 samples, 1500 max features, max iter of 100, it often doesn't 
predict anything. When it does, it also isn't always correct. Also, the longest 
training time is for the first layer, as there are many features. I believe
the max amount of features is around 15000.

Hypothesis: maybe the predictions are "bad" because we go from 2500 features to 7 
and then back up to 220. The information just cannot carry well through so little.