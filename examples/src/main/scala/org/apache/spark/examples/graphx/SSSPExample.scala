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
package org.apache.spark.examples.graphx

// $example on$
import org.apache.spark.graphx.{Graph, VertexId}
import org.apache.spark.graphx.util.GraphGenerators
// $example off$
import org.apache.spark.sql.SparkSession

/**
 * An example use the Pregel operator to express computation
  * SSSP,单源最短路径分析,对在权图G=(V,E),从一个源点s到汇点t有很多路径，其中路径上权和最少的路径，称从s到t的最短路径。简单讲：找出连接两个给定点的最低成本路径。
 * such as single source shortest path
  *
  * 对SSSP的解释：
  * Dijkstra算法是求解有向图中单源最短距离（Single Source Shortest Path，简称为SSSP）的经典算法。
  * 最短距离：对一个有权重的有向图G=(V,E)，从一个源点s到汇点v有很多路径，其中边权和最小的路径，称从s 到v的最短距离。
  * 算法基本原理，如下所示：
  * 初始化：源点s到s自身的距离（d[s]=0），其他点u到s的距离为无穷（d[u]=∞）。
  * 迭代：若存在一条从u到v的边，那么从s到v的最短距离更新为：d[v]=min(d[v], d[u]+weight(u, v))，直到所有的点到s 的距离不再发生变化时，迭代结束。
  *
  * graphx数据源问题：
  * 导入数据集方式（A：RDD方式，B:GraphLoader.edgeListFile(sc,hdfs-path），C:随机图生成器GraphGenerators）
  * https://blog.csdn.net/sinat_29508201/article/details/51679827
 * Run with
 * {{{
 * bin/run-example graphx.SSSPExample
 * }}}
 */
object SSSPExample {
  def main(args: Array[String]): Unit = {
    // Creates a SparkSession.
    val spark = SparkSession
      .builder
      .appName(s"${this.getClass.getSimpleName}")
      .getOrCreate()
    val sc = spark.sparkContext

    // $example on$
    // A graph with edge attributes containing distances
    val graph: Graph[Long, Double] =
      GraphGenerators.logNormalGraph(sc, numVertices = 100).mapEdges(e => e.attr.toDouble)
    graph.vertices.collect.foreach(println(_))
    graph.edges.collect.foreach(println(_))

    val sourceId: VertexId = 42 // The ultimate source 定义原点
    // Initialize the graph such that all vertices except the root have distance infinity.
    //初始化各节点到原点的距离
    val initialGraph = graph.mapVertices((id, _) =>
        if (id == sourceId) 0.0 else Double.PositiveInfinity)
    initialGraph.vertices.collect.foreach(println(_))
    initialGraph.edges.collect.foreach(println(_))

    val sssp = initialGraph.pregel(Double.PositiveInfinity)(
      (id, dist, newDist) => math.min(dist, newDist), // Vertex Program，节点处理消息的函数，dist为原节点属性（Double），newDist为消息类型（Double）
      triplet => {  // Send Message，发送消息函数，返回结果为（目标节点id，消息（即最短距离））
        if (triplet.srcAttr + triplet.attr < triplet.dstAttr) {
          Iterator((triplet.dstId, triplet.srcAttr + triplet.attr))
        } else {
          Iterator.empty
        }
      },
      (a, b) => math.min(a, b) // Merge Message，对消息进行合并的操作，类似于Hadoop中的combiner
    )
    println(sssp.vertices.collect.mkString("\n"))
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println

/*
输入数据格式（代码中的graph）：
vertices(vertices是无法完全描述graph的)
(84,79)
(96,38)
(52,75)
(56,53)
(4,46)
(76,26)
(16,40)
(28,36)
(80,30)
(48,88)
(32,70)
(36,19)
(0,70)
(24,95)
(64,9)
(92,31)
(40,95)
(72,88)
(8,88)
  ...


edegs：
Edge(0,5,1.0)
Edge(0,13,1.0)
Edge(0,24,1.0)
Edge(0,30,1.0)
Edge(0,36,1.0)
Edge(0,43,1.0)
Edge(0,43,1.0)
Edge(0,44,1.0)
Edge(0,44,1.0)
Edge(0,46,1.0)
Edge(0,48,1.0)
Edge(0,62,1.0)
Edge(0,66,1.0)
Edge(0,69,1.0)
Edge(0,69,1.0)
Edge(0,70,1.0)
Edge(0,72,1.0)
Edge(0,78,1.0)
Edge(0,82,1.0)
Edge(0,83,1.0)
Edge(0,85,1.0)
Edge(0,86,1.0)
Edge(0,89,1.0)
Edge(0,91,1.0)
Edge(0,93,1.0)
Edge(0,94,1.0)
Edge(0,96,1.0)
  ...


图初始化（代码中的initialGraph）:
(84,Infinity)
(96,Infinity)
(52,Infinity)
(56,Infinity)
(4,Infinity)
(76,Infinity)
(16,Infinity)
(28,Infinity)
(80,Infinity)
(48,Infinity)
(32,Infinity)
(36,Infinity)
(0,Infinity)
(24,Infinity)
(64,Infinity)
(42,0.0)
(92,Infinity)
(40,Infinity)
  ...

结果（代码中的sssp）：
(84,2.0)
(96,2.0)
(52,1.0)
(56,2.0)
(4,2.0)
(76,2.0)
(16,2.0)
(28,2.0)
(80,2.0)
(48,1.0)
(32,2.0)
(36,2.0)
(0,2.0)
(24,2.0)
(64,2.0)
(92,2.0)
(40,1.0)
(72,2.0)
(8,2.0)
(12,2.0)
(42,0.0)
  ...



如何查看图内容：
validGraph.vertices.collect.foreach(println(_))

如何进行一步一步的分析：
使用graph.edges.collect.foreach(println(_))的结果进行分析，最终得到sssp.vertices.collect.mkString("\n")
关于triples的解释：
https://endymecy.gitbooks.io/spark-graphx-source-analysis/content/vertex-edge-triple.html
核心关键：pregel
对SSSP的比较好的解释：
https://tech.antfin.com/docs/2/27907
*/
