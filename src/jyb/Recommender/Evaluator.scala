package jyb.Recommender

import org.apache.spark.rdd.RDD
import scala.collection.mutable

case class EvalTopN(k: Int) {

  def eval(test: RDD[(Int, Set[Int], Set[Int])],
           userPos: RDD[(Int, Point)],
           itemPos: Map[Int, Point]):
  (Double, Double) = {
    val testKV = test.map{
      case (u, i1, i2) => (u, (i1, i2))
    }
    val sc = test.sparkContext
    val hBD = sc.broadcast(itemPos)
    val performance = testKV.join(userPos)
      .mapValues{case ((trainItems, testItems), wu) =>
        val allItems = hBD.value.keySet
        val itemsNotInTrain = allItems.diff(trainItems)
        val itemsWithDistance = itemsNotInTrain
          .map{j =>
            val hj = hBD.value(j)
            (j, distance2(wu, hj))
          }.toArray
        val topKList = getTopKList(itemsWithDistance, this.k)
        val hits = topKList.count(testItems.contains)
        val precision = divide(hits, this.k)
        val recall = divide(hits, testItems.size)
        (precision, recall)
      }
    val nUsers = performance.count()
    val (tp, tr) = performance.map(_._2).reduce{
      case ((p0, r0), (p1, r1)) =>
        (p0 + p1, r0 + r1)
    }
    (divide(tp, nUsers), divide(tr, nUsers))
  }

  def getTopKList(scores: Array[(Int, Double)], k: Int):
  Array[Int] = {
    implicit  val ord: Ordering[(Int, Double)] =
      Ordering.by(-_._2)
    val pq = mutable.PriorityQueue[(Int, Double)]()
    scores.foreach(p => pq.enqueue(p))
    Array.fill(k)(pq.dequeue()._1)
  }
}

case class EvalRank(){
  def dcgGain(rank: Int): Double =
    divide(math.log(2.0), math.log(rank + 2))

  def getIdealDCG(k: Int): Double = {
    var loss = 0.0
    for (i <- 0 until k){
      loss += dcgGain(i)
    }
    loss
  }

  def eval(test: RDD[(Int, Set[Int], Set[Int])],
           userPos: RDD[(Int, Point)],
           itemPos: Map[Int, Point]):
  (Double, Double, Double) = {
    val testKV = test.map{
      case (u, i1, i2) => (u, (i1, i2))
    }
    val sc = test.sparkContext
    val hBD = sc.broadcast(itemPos)
    val performance = testKV.join(userPos)
      .mapValues{case ((trainItems, testItems), wu) =>
        val allItems = hBD.value.keySet
        val itemsNotInTrain = allItems.diff(trainItems)
        val itemsWithDistance = itemsNotInTrain
          .map{j =>
            val hj = hBD.value(j)
            (j, distance2(wu, hj))
          }.toArray
        val topKList = getTopKList(itemsWithDistance, testItems.size)
        val map = getMAP(topKList, testItems)
        val mrr = getMRR(topKList, testItems)
        val ndcg = getNDCG(topKList, testItems)
        (map, mrr, ndcg)
      }
    val nUsers = performance.count()
    val (map, mrr, ndcg) = performance.map(_._2).reduce{
      case ((map0, mrr0, ndcg0), (map1, mrr1, ndcg1)) =>
        (map0 + map1, mrr0 + mrr1, ndcg0 + ndcg1)
    }
    (divide(map, nUsers),
      divide(mrr, nUsers),
      divide(ndcg, nUsers))
  }

  def getMAP(topN: Array[Int], is: Set[Int]):
  Double = {
    val (map, sz) = topN.indices.foldLeft((0.0, 0)){
      case ((ap, n), idx) =>
        val j = topN(idx)
        val yij = if (is.contains(j)) 1 else 0
        val rank = idx + 1
        (ap + divide(n + yij, rank), n + yij)
    }
    divide(map, sz)
  }

  def getMRR(topN: Array[Int], is: Set[Int]):
  Double = {
    val mrr = topN.indices.foldLeft(0.0){
      case (r, idx) =>
        val j = topN(idx)
        if (is.contains(j))
          r + divide(1, idx + 1)
        else
          r
    }
    divide(mrr, is.size)
  }

  def getNDCG(topN: Array[Int], is: Set[Int]):
  Double = {
    val sz = topN.length
    val iDCG = getIdealDCG(sz)
    val DCG = topN.indices.foldLeft(0.0){
      case (dcg, idx) =>
        val j = topN(idx)
        val wij = dcgGain(idx)
        val yij = if (is.contains(j)) 1 else 0
        dcg + wij * yij
    }
    divide(DCG, iDCG)
  }

  def getTopKList(scores: Array[(Int, Double)], k: Int):
    Array[Int] = {
    implicit  val ord: Ordering[(Int, Double)] =
      Ordering.by(-_._2)
    val pq = mutable.PriorityQueue[(Int, Double)]()
    scores.foreach(p => pq.enqueue(p))
    Array.fill(k)(pq.dequeue()._1)
  }

}
