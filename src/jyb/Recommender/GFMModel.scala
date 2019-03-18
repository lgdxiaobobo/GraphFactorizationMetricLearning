package jyb.Recommender

import scala.util.Random
import scala.collection.mutable

import org.apache.spark
import spark.rdd.RDD
import spark.Partitioner
import spark.{HashPartitioner => FMPart}

case class GFMModel(setting: Setting) {
  // common setting
  private val nDim = setting.getSpaceDim
  private val eta = setting.getLearningRate
  private val alpha = setting.getLossWeight
  private val blockNum = setting.getBlockNum

  // push power
  private val negNum = setting.getNegNum
  private val margin = setting.getMargin

  // regularization related
  private val radius = setting.getClipRadius
  private val drop = setting.getDropRate

  // batch-related
  private val maxIterNum = setting.getMaxIterTimes
  private val batchSize = setting.getBatchSize
  private val batchPerIter = setting.getBatchNumPerIter

  private val epsilon = 1e-8

  // check directory
  private var checkDir = ""
  def setCheckDir(dir: String): this.type = {
    this.checkDir = dir
    this
  }

  // eval method
  // using rank-based method (ACG)
  def eval(test: RDD[(Int, Array[Int], Array[Int])],
           userPos: RDD[(Int, Point)], itemPos: Map[Int, Point]): 
  Double = {
    val sc = test.sparkContext
    val itemPosBD = sc.broadcast(itemPos)
    // union test and userPos by reduce-by-key
    val testTemp = test.map{
      case (u, is, js) => (u, (is, js))
    }
    val comb = userPos.join(testTemp)
      .mapValues{
        case (wu, (is, js)) =>
          val iPos = itemPosBD.value
          val items = iPos.keySet
          val used = (is ++ js).toSet
          val unused = items
            .diff(used).toArray
          val AG = js
            .map(gain(wu, unused, iPos))
            .sum
          divide(AG, js.length)
      }
    comb.map(_._2).mean()
  }

  // training method
  def train(train: RDD[Metric],
            test: RDD[(Int, Array[Int], Array[Int])],
            seed: Long):
  (RDD[(Int, Point)], Map[Int, Point]) = {
    // define partitioner
    val myPart = new FMPart(this.blockNum)
    // parted interactions
    val partedInteraction =
      partMetric(train, myPart)
        .persist()
    // blocked based on users
    val userBasedBlocks =
      blockedInteraction(partedInteraction, myPart)
        .persist()
    // seed generator for initialize and sampling
    val seedGen = new Random(seed)
    // initialize
    var blockedUserPos =
      initUserPos(userBasedBlocks, this.nDim, seedGen.nextLong())
    var itemPosLocal =
      initItemPos(train, this.nDim, seedGen.nextLong())
    // setting grads summation for AdaGrads
    var blockedUserSG =
      blockedUserPos.mapValues(fs => fs.map(_ => 0.0))
    var itemSG = itemPosLocal.map{
      case (k, _) =>
        (k, 0.0)
    }
    // training
    // TODO: early-stop
    for (step <- 0 until this.maxIterNum){
      val trainLoss = 
        getTrainLoss(userBasedBlocks, blockedUserPos, itemPosLocal)
      val userPos = userBasedBlocks.join(blockedUserPos)
        .mapValues{ case (b, fs) =>
          b.srcIds.zip(fs)
        }.flatMap(_._2).setName(s"$step-W").persist()
      val testLoss = 
        getTestLoss(test, userPos, itemPosLocal)
      userPos.unpersist()
      println(s"[$step-Loss] objective loss $trainLoss, ranking loss $testLoss")
      for (climb <- 0 until this.batchPerIter){
        val preBlockedUserPos =
          blockedUserPos.setName(s"climb-W")
            .persist()
        // learn gradients from sampled batches
        val (blockedUserGrads, itemGrads) = 
          learnGradient(userBasedBlocks, preBlockedUserPos, itemPosLocal, seedGen.nextLong())
        // AdaGrads for step-size
        val preBlockedUserSG = blockedUserSG
          .setName(s"$climb-SGW").persist()
        blockedUserSG = getAdaGrads(preBlockedUserSG, blockedUserGrads)
        itemSG = itemGrads.foldLeft(itemSG){
          case (agg, (i, ghi)) =>
            agg + (i -> (agg(i) + norm2(ghi)))
        }
        // adaptive update points
        blockedUserPos = 
          updatedPointsInBlock(preBlockedUserPos, blockedUserSG, blockedUserGrads)
        itemPosLocal = itemGrads.foldLeft(itemPosLocal){
          case (agg, (i, ghi)) =>
            val _eta = divide(this.eta, this.epsilon + math.sqrt(itemSG(i)))
            agg + (i -> clip(plus(agg(i), ghi, -_eta), this.radius))
        }
        // rid off dependency
        if (climb % 5 == 0){
          blockedUserPos = 
            saveAndReadPos(blockedUserPos, this.checkDir, "points")
          blockedUserSG =
            saveAndReadSG(blockedUserSG, this.checkDir, "sg")
        }
        preBlockedUserSG.unpersist()
        preBlockedUserPos.unpersist()
      }
      blockedUserPos = 
        saveAndReadPos(blockedUserPos, this.checkDir, "points")
          .persist()      
    }
    val finalUserPos = 
      userBasedBlocks.join(blockedUserPos).mapValues{
        case (b, fs) =>
          b.srcIds.zip(fs)
      }.flatMap(_._2)
    (finalUserPos, itemPosLocal)
  }

  def updatedPointsInBlock(blockedUserPos: RDD[(Int, Points)],
                           blockedUserSG: RDD[(Int, Array[Double])],
                           blockedUserGrads: RDD[(Int, Points)]):
  RDD[(Int, Points)] = {
    val blockedAdaGrads = blockedUserSG.join(blockedUserGrads)
      .mapValues{case (sgs, gs) =>
        sgs.zip(gs).map{
          case (sg, g) =>
            scaled(g, divide(this.eta, this.epsilon + math.sqrt(sg)))
        }
      }.setName("AdaGrads").persist()
    blockedAdaGrads.count()
    val updated = blockedAdaGrads.join(blockedUserPos)
      .mapValues{case (ags, ps) =>
        ps.zip(ags).map{
          case (p, ag) =>
            clip(plus(p, ag, -1.0), this.radius)
        }
      }
    updated.count()
    blockedAdaGrads.unpersist()
    updated
  }

  def getAdaGrads(blockedUserSG: RDD[(Int, Array[Double])],
                  blockedUserGrads: RDD[(Int, Points)]):
  RDD[(Int, Array[Double])] = {
    blockedUserSG.join(blockedUserGrads)
      .mapValues{case (sgs, gs) =>
        sgs.zip(gs).map{case (sg, g) =>
          sg + norm2(g)
        }
      }
  }

  def learnGradient(blocks: RDD[(Int, Block)],
                    blockedPos: RDD[(Int, Points)],
                    itemPos: Map[Int, Point], seed: Long):
  (RDD[(Int, Points)], Array[(Int, Point)]) = {
    val sc = blocks.sparkContext
    val hBD = sc.broadcast(itemPos)
    val rng = new Random(seed)
    val joint = blocks.join(blockedPos).mapValues{
      case (b, ps) =>
        val items = hBD.value.keySet.toArray
        val nItems = items.length
        val batches = 
          genBatches(b, items, this.negNum, this.batchSize, rng.nextLong())
        val gw = ps.map(p => Array.fill(p.length)(0.0))
        val ghCollector = mutable.ArrayBuilder.make[(Int, Point)]
        batches.foreach{
          case (Metric(u, i, dui), negs) =>
            val wu = ps(u)
            val hi = hBD.value(i)
            // pull with drop-out
            val tui = minus(wu, hi)
            val _tui = dropOut(tui, this.drop, rng.nextLong())
            val pullLoss = dui - norm2(_tui)
            val pullScale = -4 * this.alpha * pullLoss / this.batchSize
            // update by pull loss
            gw(u) = plus(gw(u), _tui, pullScale)
            ghCollector += ((i, scaled(_tui, -pullScale)))
            // push
            val negsPos = negs.map(hBD.value.apply)
            val (j, nk) = approRank(this.margin, wu, hi, negsPos)
            if (nk > 0) {
              val nPos = b.dstPtrs(u+1) - b.dstPtrs(u)
              val rank = ((nItems - nPos) * nk * 1.0 / this.negNum).toInt
              val kui = log2(rank + 2.0)
              val hj = hBD.value(j)
              val tuj = minus(wu, hj)
              val _tuj = dropOut(tuj, this.drop, rng.nextLong())
              val _tij = minus(_tuj, _tui)
              val pushScale = -2 * (1 - this.alpha) * kui / this.batchSize
              gw(u) = plus(gw(u), _tij, pushScale)
              ghCollector += ((i, scaled(_tui, pushScale)))
              ghCollector += ((j, scaled(_tuj, -pushScale)))
            }
        }
        val gh = ghCollector.result()
          .foldLeft(Map[Int, Point]()){
            case (agg, (i, ghi)) =>
              if (!agg.contains(i)) agg + (i -> ghi)
              else agg + (i -> plus(agg(i), ghi))
          }.toArray
        // gh.take(10).foreach{case (k, v) => println(k, v.mkString(","))}
        (gw, gh)
    }
    val blockedUserGrads = joint.mapValues(_._1)
    val itemGrads = joint.map(_._2._2)
      .flatMap(gh => gh).reduceByKey(plus)
      .collect()
    (blockedUserGrads, itemGrads)
  }

  def getTrainLoss(blocks: RDD[(Int, Block)], 
                   blockedPos: RDD[(Int, Points)], 
                   itemPos: Map[Int, Point]):
  Double = {
    val sc = blocks.sparkContext
    val hBD = sc.broadcast(itemPos)
    val errors = blocks.join(blockedPos)
      .mapValues{case (b, ps) =>
        val ptr = b.dstPtrs
        val dstIds = b.dstIds
        val distances = b.distances
        val sz = b.sz
        val eInB = Array.fill(sz)(0.0)
        for (u <- 0 until sz){
          val wu = ps(u)
          val left = ptr(u)
          val right = ptr(u+1)
          val push =
            pushLoss(this.margin, wu, left, right, dstIds, hBD.value)
          val pull =
            pullLoss(wu, left, right, dstIds, distances, hBD.value)
          eInB(u) = this.alpha * pull + (1 - this.alpha) * push
        }
        eInB
      }
    errors.flatMap(_._2).mean()
  }

  def getTestLoss(test: RDD[(Int, Array[Int], Array[Int])],
                  userPos: RDD[(Int, Point)], itemPos: Map[Int, Point]):
  Double = 
    this.eval(test, userPos, itemPos)

  // change u-i interaction as block-item
  def blockedInteraction(interaction: RDD[(Int, Metrics)],
                         part: Partitioner):
  RDD[(Int, Block)] = {
    interaction.mapPartitions({items =>
      items.map{case (idx, ms) =>
        val builder = BlockBuilder()
        builder.add(ms)
        (idx, builder.build())
      }
    }, preservesPartitioning = true)
      .setName("UserBlock")
  }

  // part u-i interaction based on user
  def partMetric(data: RDD[Metric],
                 part: Partitioner):
  RDD[(Int, Metrics)] = {
    val numPartitions = part.numPartitions
    data.mapPartitions({ps =>
      val builders =
        Array.fill(numPartitions)(MetricsBuilder())
      ps.flatMap{r =>
        val blockId = part.getPartition(r.src)
        val builder = builders(blockId)
        builder.add(r)
        if (builder.size >= 1024){
          builders(blockId) = MetricsBuilder()
          Iterator.single((blockId, builder.build()))
        }else Iterator.empty
      } ++ {
        builders.view.zipWithIndex
          .filter(_._1.size > 0).map{
          case (block, idx) =>
            (idx, block.build())
        }
      }
    }, preservesPartitioning = true)
      .groupByKey(numPartitions).mapValues{ps =>
      val builder = MetricsBuilder()
      ps.foreach(builder.merge)
      builder.build()
    }.setName("BlockedMetrics")
  }

}
