dataframe <- read.csv("/home/data/MT/dataset3/d21.csv" , header = F , sep = ",");
subset <- c (3:12,15,17,18,19,20,21,22,22,38,43,44,45,46,47);
dataframeSubset <- dataframe[subset];
aggregate(V3~V19,dataframeSubset,mean);
aggregate(cbind(V4,V8) ~ V44, dataframeSubset , mean);
