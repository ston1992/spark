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

/*
*特征缩放
* 因为在原始的资料中，各变数的范围大不相同。对于某些机器学习的算法，若没有做过标准化，目标函数会无法适当的运作。
* 举例来说，多数的分类器利用两点间的距离计算两点的差异， 若其中一个特征具有非常广的范围，那两点间的差异就会被该特征左右，因此，所有的特征都该被标准化，这样才能大略的使各特征依比例影响距离。
* 另外一个做特征缩放的理由是他能使加速梯度下降法的收敛。
*/

// $example on$
import org.apache.spark.ml.feature.StandardScaler
// $example off$
import org.apache.spark.sql.SparkSession

object StandardScalerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("StandardScalerExample")
      .getOrCreate()

    // $example on$
    val dataFrame = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(true)
      .setWithMean(false)

    // Compute summary statistics by fitting the StandardScaler.
    val scalerModel = scaler.fit(dataFrame)

    // Normalize each feature to have unit standard deviation.
    val scaledData = scalerModel.transform(dataFrame)
    scaledData.show()
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
