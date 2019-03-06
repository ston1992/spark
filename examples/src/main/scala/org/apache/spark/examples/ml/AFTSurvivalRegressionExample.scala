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
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.AFTSurvivalRegression
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * An example demonstrating AFTSurvivalRegression.
  * 加速失效时间模型（Acceleratedfailure time）
  * 加速失效（accelerate failure time）模型的假设是，一个人的生存时间等于人群基准生存时间 * 这个人的加速因子。
  * ！！！！可以用来进行不同行业的不同标签的有效时间的计算
 * Run with
 * {{{
 * bin/run-example ml.AFTSurvivalRegressionExample
 * }}}
 */
object AFTSurvivalRegressionExample {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("AFTSurvivalRegressionExample")
      .getOrCreate()

    // $example on$
    val training = spark.createDataFrame(Seq(
      (1.218, 1.0, Vectors.dense(1.560, -0.605)),
      (2.949, 0.0, Vectors.dense(0.346, 2.158)),
      (3.627, 0.0, Vectors.dense(1.380, 0.231)),
      (0.273, 1.0, Vectors.dense(0.520, 1.151)),
      (4.199, 0.0, Vectors.dense(0.795, -0.226))
    )).toDF("label", "censor", "features")
    val quantileProbabilities = Array(0.3, 0.6)
    val aft = new AFTSurvivalRegression()
      .setQuantileProbabilities(quantileProbabilities)
      .setQuantilesCol("quantiles")

    val model = aft.fit(training)

    // Print the coefficients, intercept and scale parameter for AFT survival regression
    println(s"Coefficients: ${model.coefficients}")
    println(s"Intercept: ${model.intercept}")
    println(s"Scale: ${model.scale}")
    model.transform(training).show(false)
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println

/*
结果
+-----+------+--------------+------------------+---------------------------------------+
|label|censor|features      |prediction        |quantiles                              |
+-----+------+--------------+------------------+---------------------------------------+
|1.218|1.0   |[1.56,-0.605] |5.7189794876349636|[1.1603238947151586,4.995456010274733] |
|2.949|0.0   |[0.346,2.158] |18.07652118149563 |[3.667545845471803,15.789611866277887] |
|3.627|0.0   |[1.38,0.231]  |7.381861804239099 |[1.4977061305190849,6.447962612338964] |
|0.273|1.0   |[0.52,1.151]  |13.57761250142538 |[2.7547621481507076,11.859872224069786]|
|4.199|0.0   |[0.795,-0.226]|9.013097744073843 |[1.8286676321297732,7.872826505878383] |
+-----+------+--------------+------------------+---------------------------------------+
*/
