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
* IndexToString maps a column of label indices back to a column containing the original labels as strings.
* StringIndexer是指把一组字符型标签编码成一组标签索引，索引的范围为0到标签数量，索引构建的顺序为标签的频率，优先编码频率较大的标签，所以出现频率最高的标签为0号。如果输入的是数值型的，我们会把它转化成字符型，然后再对其进行编码。
* 这个算法在搜索等需要创建索引的场景下很实用
*/

// $example on$
import org.apache.spark.ml.attribute.Attribute
import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
// $example off$
import org.apache.spark.sql.SparkSession

object IndexToStringExample {
  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("IndexToStringExample")
      .getOrCreate()

    // $example on$
    val df = spark.createDataFrame(Seq(
      (0, "a"),
      (1, "b"),
      (2, "c"),
      (3, "a"),
      (4, "a"),
      (5, "c"),
      (6, "c"),
      (7, "c"),
      (8, "c")
    )).toDF("id", "category")

    val indexer = new StringIndexer()
      .setInputCol("category")
      .setOutputCol("categoryIndex")
      .fit(df)
    val indexed = indexer.transform(df)

    println(s"Transformed string column '${indexer.getInputCol}' " +
        s"to indexed column '${indexer.getOutputCol}'")
    indexed.show()

    val inputColSchema = indexed.schema(indexer.getOutputCol)
    println(s"StringIndexer will store labels in output column metadata: " +
        s"${Attribute.fromStructField(inputColSchema).toString}\n")

    val converter = new IndexToString()
      .setInputCol("categoryIndex")
      .setOutputCol("originalCategory")

    val converted = converter.transform(indexed)

    println(s"Transformed indexed column '${converter.getInputCol}' back to original string " +
        s"column '${converter.getOutputCol}' using labels in metadata")
    converted.select("id", "categoryIndex", "originalCategory").show()
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println

/*
result:
+---+--------+-------------+
| id|category|categoryIndex|
+---+--------+-------------+
|  0|       a|          1.0|
|  1|       b|          2.0|
|  2|       c|          0.0|
|  3|       a|          1.0|
|  4|       a|          1.0|
|  5|       c|          0.0|
|  6|       c|          0.0|
|  7|       c|          0.0|
|  8|       c|          0.0|
+---+--------+-------------+

+---+-------------+----------------+
| id|categoryIndex|originalCategory|
+---+-------------+----------------+
|  0|          1.0|               a|
|  1|          2.0|               b|
|  2|          0.0|               c|
|  3|          1.0|               a|
|  4|          1.0|               a|
|  5|          0.0|               c|
|  6|          0.0|               c|
|  7|          0.0|               c|
|  8|          0.0|               c|
+---+-------------+----------------+
*/
