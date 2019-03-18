package jyb.Recommender

import org.apache.spark.rdd.RDD

case class Performance(map: Double,
                       mrr: Double,
                       ndcg: Double){
  def add(elem: Performance): Performance = {
    Performance(map + elem.map,
      mrr + elem.mrr, ndcg + elem.ndcg)
  }

  def scaled(w: Double): Performance =
    Performance(map / w, mrr / w, ndcg / w)

  override def toString: String =
    s"MAP:$map, MRR:$mrr, nDCG:$ndcg"
}

case class Evaluator(k: Int) {

  private val iDCG = getIdealDCG()
  private val log2 = math.log(2.0)

  def dcgGain(rank: Int): Double =
    this.log2 / math.log(rank + 2)

  def getIdealDCG(): Double = {
    var loss = 0.0
    for (i <- 0 until this.k){
      loss += dcgGain(i)
    }
    loss
  }

  def eval(test: RDD[(Int, Array[Int], Array[Int])],
           userPos: RDD[(Int, Point)],
           itemPos: Map[Int, Point]):
  Performance = {
    val testKV = test.map{
      case (u, i1, i2) => (u, (i1, i2))
    }
    val sc = test.sparkContext
    val hBD = sc.broadcast(itemPos)
    val performance = testKV.join(userPos)
      .mapValues{case ((i1, i2), wu) =>
        val allItems = hBD.value.keySet
        val trainItems = i1.toSet
        val testItems = i2.toSet
        val itemsNotInTrain = allItems.diff(trainItems)
        val itemsWithDistance = itemsNotInTrain
          .map{j =>
            val hj = hBD.value(j)
            (j, distance2(wu, hj))
          }.toArray
        val topKList = getTopKList(itemsWithDistance)
        val nDCG = getNDCG(topKList, testItems)
        val map = getMap(topKList, testItems)
        val mrr = getMRR(topKList, testItems)
        Performance(nDCG, map, mrr)
      }
    val nUsers = performance.count()
    val total = performance.map(_._2).reduce(_ add _)
    total.scaled(nUsers)
  }

  def getMRR(topK: Array[Int],
             is: Set[Int]):
  Double = {
    topK.zipWithIndex.foldLeft(0.0){
      case (loss, (j, rank)) =>
        if (!is.contains(j))
          loss
        else
          loss + 1.0 / (1.0 + rank)
    } / topK.length
  }

  def getMap(topK: Array[Int],
             is: Set[Int]):
  Double = {
    val hits = topK.count(is.contains)
    hits * 1.0 / topK.length //is.size
  }

  def getNDCG(topK: Array[Int],
              is: Set[Int]):
  Double = {
    val DCG = topK.zipWithIndex.foldLeft(0.0){
      case (loss, (j, rank)) =>
        if (!is.contains(j))
          loss
        else
          loss + dcgGain(rank)
    }
    DCG / iDCG
  }

  def getTopKList(scores: Array[(Int, Double)]):
  Array[Int] = {
    scores.sortBy(_._2).map(_._1).take(this.k)
  }
}
