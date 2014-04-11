log-regression
==============

This script classifies music files among one of the different music genres
[classical, country, jazz, reggae, rock, metal]. This script uses 10-fold 
cross validation to generate the training set and the validation set from
the data set.

- To run this script it is necessary to have installed python 2.7.X and Scipy.

- The data will be retrieved from diffetent txt files:
  classes.txt          -> contains the genre of each song
  features.txt         -> conitains the FFT features of every song
  features_mfcc.txt    -> contains the MFCC features of every song
  features-120-fft.txt -> contains the top 120 FFT features of every song 

- The script must be run with one parameter from console:
  python classifier.py -f -> to learn with the FFT features
  python classifier.py -m -> to learn with the MFCC features
  python classifier.py -r -> to learn with the top 120 FFT features