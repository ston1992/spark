/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark.examples.ml

// $example on$
import org.apache.spark.ml.feature.BucketedRandomProjectionLSH
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
// $example off$

/**
 * An example demonstrating BucketedRandomProjectionLSH.
  * LSH是大规模机器学习中常用的随机算法和哈希技术，包括聚类和近似最近邻搜索。
  * 实现 LSH 之前，Uber筛选行程的算法复杂度为 N^2，虽然精度很高，但是太耗费资源。LSH的总体思路是使用一系列函数（称为 LSH 族）将数据点哈希到桶(buckets)中，使距离较近的数据点位于同一个桶中的概率较高，而距离很远的数据点在不同的桶里。因此, LSH 算法能使具有不同程度重叠行程的识别更为容易。
  * 在Spark 2.1中，有两个LSH估计器：
  *1、 基于欧几里德距离的BucketedRandomProjectionLSH【它研究的对象是向量，向量是有方向的。】
  *2、 基于Jaccard距离的MinHashLSH   【它研究的对象，其实是集合而不是向量，集合没有方向。】
    * Jaccard相似指数用来度量两个集合之间的相似性，它被定义为两个集合交集的元素个数除以并集的元素个数。
    * Jaccard距离用来度量两个集合之间的差异性，它是Jaccard的相似系数的补集，被定义为1减去Jaccard相似系数。
    * https://baike.baidu.com/item/%E6%9D%B0%E5%8D%A1%E5%BE%B7%E8%B7%9D%E7%A6%BB/15416212?fr=aladdin
    * 关于各种距离概念的介绍：https://blog.csdn.net/mpk_no1/article/details/72935442
  * 大量应用包括：
  * 》近似重复的检测： LSH 通常用于对大量文档，网页和其他文件进行去重处理。
  * 》全基因组的相关研究：生物学家经常使用 LSH 在基因组数据库中鉴定相似的基因表达。
  * 》大规模的图片搜索： Google 使用 LSH 和 PageRank 来构建他们的图片搜索技术VisualRank。
  * 》音频/视频指纹识别：在多媒体技术中，LSH 被广泛用于 A/V 数据的指纹识别。
  * Run with:
 *   bin/run-example ml.BucketedRandomProjectionLSHExample
  **/
object BucketedRandomProjectionLSHExample {
  def main(args: Array[String]): Unit = {
    // Creates a SparkSession
    val spark = SparkSession
      .builder
      .appName("BucketedRandomProjectionLSHExample")
      .getOrCreate()

    // $example on$
    val dfA = spark.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 1.0)),
      (1, Vectors.dense(1.0, -1.0)),
      (2, Vectors.dense(-1.0, -1.0)),
      (3, Vectors.dense(-1.0, 1.0))
    )).toDF("id", "features")

    val dfB = spark.createDataFrame(Seq(
      (4, Vectors.dense(1.0, 0.0)),
      (5, Vectors.dense(-1.0, 0.0)),
      (6, Vectors.dense(0.0, 1.0)),
      (7, Vectors.dense(0.0, -1.0))
    )).toDF("id", "features")

    val key = Vectors.dense(1.0, 0.0)

    val brp = new BucketedRandomProjectionLSH()
      .setBucketLength(2.0)
      .setNumHashTables(3)
      .setInputCol("features")
      .setOutputCol("hashes")

    val model = brp.fit(dfA)

    // Feature Transformation
    println("The hashed dataset where hashed values are stored in the column 'hashes':")
    model.transform(dfA).show()

    // Compute the locality sensitive hashes for the input rows, then perform approximate
    // similarity join.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxSimilarityJoin(transformedA, transformedB, 1.5)`
    println("Approximately joining dfA and dfB on Euclidean distance smaller than 1.5:")
    model.approxSimilarityJoin(dfA, dfB, 1.5, "EuclideanDistance")
      .select(col("datasetA.id").alias("idA"),
        col("datasetB.id").alias("idB"),
        col("EuclideanDistance")).show()

    // Compute the locality sensitive hashes for the input rows, then perform approximate nearest
    // neighbor search.
    // We could avoid computing hashes by passing in the already-transformed dataset, e.g.
    // `model.approxNearestNeighbors(transformedA, key, 2)`
    println("Approximately searching dfA for 2 nearest neighbors of the key:")
    model.approxNearestNeighbors(dfA, key, 2).show()
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
