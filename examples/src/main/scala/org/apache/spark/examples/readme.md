###### 1、ML是基于DATAFRAME，官方推荐；MLlib是基于RDD，可能在spark3.0中被废弃。

spark2-shell --executor-memory 5g --driver-memory 1g --master spark://cdh-node02:7077 < PageRankExample.scala