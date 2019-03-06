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
* StringIndexer是指把一组字符型标签编码成一组标签索引，索引的范围为0到标签数量，索引构建的顺序为标签的频率，优先编码频率较大的标签，所以出现频率最高的标签为0号。如果输入的是数值型的，我们会把它转化成字符型，然后再对其进行编码。
*/

// $example on$
import org.apache.spark.ml.feature.StringIndexer
// $example off$
import org.apache.spark.sql.SparkSession

object StringIndexerExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("StringIndexerExample")
      .getOrCreate()

    // $example on$
    val df = spark.createDataFrame(
      Seq((0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"))
    ).toDF("id", "category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")

    val indexed = indexer.fit(df).transform(df)
    indexed.show()
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
