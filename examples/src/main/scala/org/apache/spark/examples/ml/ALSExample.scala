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
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * An example demonstrating ALS.
  * ALS：协同过滤算法。目标在于填充“用户－商品”联系矩阵中的缺失项。目前，仅支持整数型的用户和商品编号。【在spark ML中是隐式反馈】
  * 显式与隐式反馈：基于矩阵分解的协同过滤的标准方法中，“用户－商品”矩阵中的条目是用户给予商品的显式偏好，例如，用户给电影评级。然而在现实世界中使用时，我们常常只能访问隐式反馈（如意见、点击、购买、喜欢以及分享等），在spark.ml中我们使用“隐式反馈数据集的协同过滤“来处理这类数据。本质上来说它不是直接对评分矩阵进行建模，而是将数据当作数值来看待，这些数值代表用户行为的观察值（如点击次数，用户观看一部电影的持续时间）。这些数值被用来衡量用户偏好观察值的置信水平，而不是显式地给商品一个评分。然后，模型用来寻找可以用来预测用户对商品预期偏好的潜在因子。
 * 调整正则化参数regParam来解决用户在更新用户因子时产生新评分或者商品更新商品因子时收到的新评分带来的最小二乘问题。这个方法叫做“ALS-WR”它降低regParam对数据集规模的依赖，所以我们可以将从部分子集中学习到的最佳参数应用到整个数据集中时获得同样的性能。
  * Run with
 * {{{
 * bin/run-example ml.ALSExample
 * }}}
 */
object ALSExample {

  // $example on$
  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }
  // $example off$

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("ALSExample")
      .getOrCreate()
    import spark.implicits._

    // $example on$
    val ratings = spark.read.textFile("data/mllib/als/sample_movielens_ratings.txt")
      .map(parseRating)
      .toDF()
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

    // Build the recommendation model using ALS on the training data
    val als = new ALS()
      .setMaxIter(5)
      .setRegParam(0.01)
      .setUserCol("userId")
      .setItemCol("movieId")
      .setRatingCol("rating")
    val model = als.fit(training)

    // Evaluate the model by computing the RMSE on the test data
    // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
    model.setColdStartStrategy("drop")
    val predictions = model.transform(test)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")
    val rmse = evaluator.evaluate(predictions)
    println(s"Root-mean-square error = $rmse")

    // Generate top 10 movie recommendations for each user
    val userRecs = model.recommendForAllUsers(10)
    // Generate top 10 user recommendations for each movie
    val movieRecs = model.recommendForAllItems(10)

    // Generate top 10 movie recommendations for a specified set of users
    val users = ratings.select(als.getUserCol).distinct().limit(3)
    val userSubsetRecs = model.recommendForUserSubset(users, 10)
    // Generate top 10 user recommendations for a specified set of movies
    val movies = ratings.select(als.getItemCol).distinct().limit(3)
    val movieSubSetRecs = model.recommendForItemSubset(movies, 10)
    // $example off$
    userRecs.show()
    movieRecs.show()
    userSubsetRecs.show()
    movieSubSetRecs.show()

    spark.stop()
  }
}
// scalastyle:on println

