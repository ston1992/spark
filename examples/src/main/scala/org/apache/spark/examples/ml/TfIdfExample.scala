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
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
// $example off$
import org.apache.spark.sql.SparkSession

object TfIdfExample {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("TfIdfExample")
      .getOrCreate()

    // $example on$
    val sentenceData = spark.createDataFrame(Seq(
      (0.0, "Hi I heard about Spark"),
      (0.0, "I wish Java could use case classes"),
      (1.0, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")  //对象为汉语时需要改变分词的逻辑，或者说，直接把分词后的对象作为第一步的处理对象（即不再需要分词）
    val wordsData = tokenizer.transform(sentenceData)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)

    val featurizedData = hashingTF.transform(wordsData)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)

    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("label", "features").show()
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println

/*
结果
+-----+--------------------+
|label|            features|
+-----+--------------------+
|  0.0|(20,[0,5,9,17],[0...|
|  0.0|(20,[2,7,9,13,15]...|
|  1.0|(20,[4,6,13,15,18...|
+-----+--------------------+
如果show words和rawFeatures
+-----+--------------------+--------------------+--------------------+
|label|               words|         rawFeatures|            features|
+-----+--------------------+--------------------+--------------------+
|  0.0|[hi, i, heard, ab...|(20,[0,5,9,17],[1...|(20,[0,5,9,17],[0...|
|  0.0|[i, wish, java, c...|(20,[2,7,9,13,15]...|(20,[2,7,9,13,15]...|
|  1.0|[logistic, regres...|(20,[4,6,13,15,18...|(20,[4,6,13,15,18...|
+-----+--------------------+--------------------+--------------------+
*/
