package jyb

/*
 define the basic functions in our euclidean space
 1. position as Array[Double]
 2. distance => Euclidean distance
 3. clip in a given radius sphere
 4. adding drop-out layer while training
 5. generate batches for training
*/

import org.apache.spark.rdd.RDD

import scala.util.Random
import scala.collection.mutable

package object Recommender {

  def divide(a: Double, b: Double):
  Double = {
    if (b == 0.0) 0.0
    else a * 1.0 / b
  }

  def gain(wu: Point, js: Array[Int],
           iPos: Map[Int, Point])(i: Int):
  Double = {
    val hi = iPos(i)
    val dui = distance2(wu, hi)
    val rank = js.map{j =>
      val hj = iPos(j)
      distance2(wu, hj) - dui
    }.count(_ <= 0)
    divide(1.0, log2(rank + 2.0))
  }

  type Point = Array[Double]
  type Points = Array[Point]

  def plus(p1: Point, w1: Double,
           p2: Point, w2: Double):
  Point = {
    require(p1.length == p2.length)
    p1.zip(p2).map{
      case (x1, x2) =>
        x1 * w1 + x2 * w2
    }
  }

  def plus(p1: Point, p2: Point,
           w2: Double):
  Point =
    plus(p1, 1.0, p2, w2)

  def plus(p1: Point, p2: Point):
  Point =
    plus(p1, 1.0, p2, 1.0)

  def minus(p1: Point, p2: Point):
  Point =
    plus(p1, 1.0, p2, -1.0)


  def distance2(p1: Point, p2: Point):
  Double = {
    require(p1.length == p2.length)
    p1.zip(p2).map{
      case (x1, x2) =>
        pow2(x1 - x2)
    }.sum
  }

  def distance(p1: Point, p2: Point):
  Double = {
    math.sqrt(distance2(p1, p2))
  }

  def norm2(p: Point):
  Double = {
    p.map(pow2).sum
  }

  def pow2(x: Double): Double =
    x * x

  def log2(x: Double): Double =
    math.log(x) / math.log(2.0)

  def div(x: Double, y: Double):
  Double =
    if (y == 0) 0.0
    else 1.0 * x / y

  def scaled(p: Point, w: Double):
  Point =
    p.map(x => x * w)

  def clip(p: Point, radius: Double):
  Point = {
    val p2o = math.sqrt(norm2(p))
    if (p2o <= radius)
      p
    else
      scaled(p, radius / p2o)
  }

  case class Metric(src: Int, dst: Int,
                    distance: Double)
  case class Metrics(srcIds: Array[Int],
                     dstIds: Array[Int],
                     distances: Array[Double]){
    val sz = srcIds.length
    require(dstIds.length == sz)
    require(distances.length == sz)
  }
  case class MetricsBuilder(){
    private val srcIdsBuilder =
      mutable.ArrayBuilder.make[Int]
    private val dstIdsBuilder =
      mutable.ArrayBuilder.make[Int]
    private val distancesBuilder =
      mutable.ArrayBuilder.make[Double]

    var size = 0

    def add(r: Metric): this.type = {
      size += 1
      srcIdsBuilder += r.src
      dstIdsBuilder += r.dst
      distancesBuilder += r.distance
      this
    }

    def merge(other: Metrics): this.type = {
      size += other.sz
      srcIdsBuilder ++= other.srcIds
      dstIdsBuilder ++= other.dstIds
      distancesBuilder ++= other.distances
      this
    }

    def build(): Metrics = {
      Metrics(srcIdsBuilder.result(),
        dstIdsBuilder.result(),
        distancesBuilder.result())
    }

  }

  case class Block(srcIds: Array[Int],
                   dstIds: Array[Int],
                   dstPtrs: Array[Int],
                   distances: Array[Double]){
    val sz = srcIds.length
    require(dstPtrs.length == sz + 1)
    require(dstPtrs.last == dstIds.length)
    require(distances.length == dstIds.length)
  }

  case class BlockBuilder(){
    private val srcIdsCollector =
      mutable.ArrayBuilder.make[Int]
    private val dstIdsCollector =
      mutable.ArrayBuilder.make[Int]
    private val distanceCollector =
      mutable.ArrayBuilder.make[Double]

    def add(elem: Metrics): this.type = {
      this.srcIdsCollector ++= elem.srcIds
      this.dstIdsCollector ++= elem.dstIds
      this.distanceCollector ++= elem.distances
      this
    }

    def build(): Block = {
      val dupSrcIds =
        this.srcIdsCollector.result()
      val dupDstIds =
        this.dstIdsCollector.result()
      val dupDistances =
        this.distanceCollector.result()

      val sortedSrcIds =
        dupSrcIds.zipWithIndex
          .sortBy(_._1)

      val uniqSrcIdsBuilder =
        mutable.ArrayBuilder.make[Int]
      val dstPtrsBuilder =
        mutable.ArrayBuilder.make[Int]
      val dstIdsBuilder =
        mutable.ArrayBuilder.make[Int]
      val distancesBuilder =
        mutable.ArrayBuilder.make[Double]
      val lastIndex = sortedSrcIds.foldLeft((-1, 0)){
        case ((preSrcId, sz), (srcId, dupIdx)) =>
          dstIdsBuilder += dupDstIds(dupIdx)
          distancesBuilder += dupDistances(dupIdx)
          if (preSrcId == -1 || preSrcId != srcId){
            uniqSrcIdsBuilder += srcId
            dstPtrsBuilder += sz
            (srcId, sz + 1)
          }else{
            (preSrcId, sz + 1)
          }
      }._2
      dstPtrsBuilder += lastIndex
      Block(uniqSrcIdsBuilder.result(),
        dstIdsBuilder.result(), dstPtrsBuilder.result(),
        distancesBuilder.result())
    }
  }

  // get x ~ N(mu, sigma^2)
  def gaussian(rng: Random, mu: Double, sigma: Double):
  Double = 
    rng.nextGaussian() * sigma + mu
  // randomly get point
  def randPoint(rng: Random, nDim: Int,
                mu: Double, sigma: Double):
  Point =
    Array.fill(nDim)(gaussian(rng, mu, sigma))

  // initialize position in a small sphere
  def initUserPos(blocks: RDD[(Int, Block)],
                  nDim: Int, seed: Long):
  RDD[(Int, Points)] = {
    val rng = new Random(seed)
    blocks.mapValues{p =>
      Array.fill(p.sz)(randPoint(rng, nDim, 0.0, 0.1))
    }
  }
  def initItemPos(X: RDD[Metric],
                  nDim: Int, seed: Long):
  Map[Int, Point] = {
    val rng = new Random(seed)
    X.map(_.dst).distinct()
     .map(i => (i, randPoint(rng, nDim, 0.0, 0.1)))
     .collect().toMap
  }

  def getIndices(indices: Array[Int],
                  l: Int, r: Int):
  Array[Int] = 
    indices.slice(l, r)
  

  def pushLoss(m: Double, wu: Point, l: Int, 
               r: Int, indices: Array[Int], 
               itemPos: Map[Int, Point]):
  Double = {
    val posItems = getIndices(indices, l, r)
    val usedSet = posItems.toSet
    val negItems = itemPos.keySet
      .diff(usedSet).toArray
    val negDistances = negItems
      .map{ j =>
        val hj = itemPos(j)
        distance2(wu, hj) - m
      }
    val sz = posItems.length
    val loss = posItems.map{i =>
      val hi = itemPos(i)
      val dui = distance2(wu, hi)
      val rank = negDistances
        .count(_ <= dui)
      if (rank == 0) 0.0
      else{
        val kui = log2(rank + 2.0)
        val maxPush = dui - negDistances.min
        kui * maxPush
      }
    }.sum / sz
    loss
  }

  def pullLoss(wu: Point, l: Int, r: Int,
               indices: Array[Int],
               distances: Array[Double],
               itemPos: Map[Int, Point]):
  Double = {
    var errors = 0.0
    val sz = r - l
    for (idx <- l until r){
      val i = indices(idx)
      val hi = itemPos(i)
      val _dui = distances(idx)
      errors += pow2(_dui - distance2(wu, hi))
    }
    errors * 1.0 / sz
  }

  def genBatches(block: Block, items: Array[Int],
                 negNum: Int, bSize: Int, seed: Long):
  Array[(Metric, Array[Int])] = {
    val nUsers = block.sz
    val rng = new Random(seed)
    val ptrs = block.dstPtrs
    val indices = block.dstIds
    val userUsedMap = ptrs.indices.dropRight(1)
      .foldLeft(Map[Int, Set[Int]]()){
        case (dict, u) =>
          val used =
            getIndices(indices, ptrs(u), ptrs(u+1))
          dict + (u -> used.toSet)
      }
    Array.fill(bSize){
      val u = rng.nextInt(nUsers)
      val usedSet = userUsedMap(u)
      val usedSz = usedSet.size
      val idx = ptrs(u) + rng.nextInt(usedSz)
      val i = indices(idx)
      val dui = block.distances(idx)
      val negItems =
        items.filter(j => !usedSet.contains(j))
      val negSize = negItems.length
      val js =
        Array.fill(negNum)(negItems.apply(rng.nextInt(negSize)))
      (Metric(u, i, dui), js)
    }
  }

  def dropOut(delta: Point, ratio: Double, seed: Long):
  Point = {
    val rng = new Random(seed)
    delta.map{xi => 
      val pi = rng.nextDouble()
      if (pi < ratio) 0.0
      else xi
    }
  }

  def approRank(margin: Double,
                wu: Point, hi: Point,
                negsPos: Points):
  (Int, Int) = {
    val dui = distance2(wu, hi)
    val negsDistances = 
      negsPos.map(hj =>
        margin + dui - distance2(wu, hj)
      )
    val nk = negsDistances.count(_ >= 0)
    val maxJ = negsDistances.tail.zipWithIndex
      .foldLeft((negsDistances.head, 0)){
        case ((maxV, maxI), (v, idx)) =>
          if (maxV >= v)
            (maxV, maxI)
          else
            (v, idx + 1)
      }._2
    (maxJ, nk)  
  }

  def saveAndReadPos(blockedPos: RDD[(Int, Points)],
                     baseDir: String, name: String):
  RDD[(Int, Points)] = {
    val partNum = blockedPos.getNumPartitions
    val sc = blockedPos.sparkContext
    val resultDir = jyb.concatPath(baseDir, name)
    jyb.deleteIfExists(sc, resultDir, true)
    blockedPos.flatMap{case (idx, ps) =>
      ps.zipWithIndex.map{case (p, idy) =>
        s"$idx|$idy|" + p.map(_.toString).mkString("|")
      }
    }.repartition(partNum).saveAsTextFile(resultDir)
    sc.textFile(resultDir).map(_.split('|'))
      .map{ps =>
        val idx = ps(0).toInt
        val idy = ps(1).toInt
        val point = ps.drop(2).map(_.toDouble)
        (idx, (idy, point))
      }.groupByKey(partNum).mapValues{ps =>
        ps.toArray.sortBy(_._1).map(_._2)
      }
  }

  def saveAndReadSG(blockedSG: RDD[(Int, Array[Double])],
                    baseDir: String, name: String):
  RDD[(Int, Array[Double])] = {
    val partNum = blockedSG.getNumPartitions
    val sc = blockedSG.sparkContext
    val resultDir = jyb.concatPath(baseDir, name)
    jyb.deleteIfExists(sc, resultDir, true)
    blockedSG.flatMap{case (idx, ps) =>
      ps.zipWithIndex.map{case (p, idy) =>
        s"$idx|$idy|$p"
      }
    }.repartition(partNum).saveAsTextFile(resultDir)
    sc.textFile(resultDir).map(_.split('|'))
      .map{ps =>
        val idx = ps(0).toInt
        val idy = ps(1).toInt
        val sg = ps(2).toDouble
        (idx, (idy, sg))
      }.groupByKey(partNum).mapValues{ps =>
      ps.toArray.sortBy(_._1).map(_._2)
    }
  }
}
