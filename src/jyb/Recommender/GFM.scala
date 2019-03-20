package jyb.Recommender

import org.apache.spark
import spark.rdd.RDD

import scala.util.Random

object GFM {

  def main(args: Array[String]): Unit = {
    // hadoop directory
    val trainDir = args(0)
    val testDir = args(1)
    val tempDir = args(2)
    // local directory
    val cfgDir = args(3)
    // position directory
    val posDir = args(4)
    // seed setting
    val givenSeed = args(5).toLong

    val ss = jyb.getSS("GFM")
    val sc = ss.sparkContext
    // correct IO setting
    require(jyb.isDirExists(sc, trainDir), s"Empty Directory for $trainDir")
    require(jyb.isDirExists(sc, testDir), s"Empty Directory for $testDir")
    require(jyb.isLocalFileExist(cfgDir), s"No setting files!")
    // load data
    // RDD[Metric(u, i, d)]
    // d = smax - s
    val train0 = sc.textFile(trainDir)
      .map(_.split('|'))
      .map(formatToDistance)
    val minD = train0.map(_.distance).min()
    val train = train0.map{p =>
      val d1 = p.distance - minD
      Metric(p.src, p.dst, d1)
    }
    println(train.map(_.distance).stats())
    val test = sc.textFile(testDir)
      .map(_.split('|'))
      .map(formatToUsage)

    /*
    val train = sc.parallelize(
      Array(
        Metric(0, 0, 0.2),
        Metric(0, 2, 0.5),
        Metric(1, 0, 0.8),
        Metric(1, 1, 0.4),
        Metric(2, 2, 0.3),
        Metric(3, 1, 0.6),
        Metric(3, 3, 0.2),
        Metric(4, 2, 0.1),
        Metric(5, 0, 0.1),
        Metric(5, 3, 0.7),
        Metric(6, 2, 0.5)
      )
    )

    val test = sc.parallelize(
      Array(
        jyb.Usage(0, 3),
        jyb.Usage(1, 3),
        jyb.Usage(2, 1),
        jyb.Usage(3, 0),
        jyb.Usage(4, 1),
        jyb.Usage(5, 2),
        jyb.Usage(6, 2)
      )
    )*/
    /*
      1. since we don't consider cold-start problem in this work,
         we will simply filtered items in test-set but not in train-set
      2. for simplify, we will add used items for users in test-set
    */
    val testWithUsedItemsDir = jyb.concatPath(tempDir, "testWithUsedItems")
    val testWithUsedItems = formatTest(train, test, testWithUsedItemsDir)
    // define GFM Model with setting
    val modelSetting = Setting(cfgDir)
    val model = GFMModel(modelSetting)
      .setCheckDir(tempDir)
    val rng = new Random(givenSeed)
    val (userPosition, itemPosition) =
      model.train(train, testWithUsedItems, rng.nextLong())
    val loss = model.eval(testWithUsedItems, userPosition, itemPosition)
    println(s"performance in test-set is $loss")
    // evaluator
    Array(5, 10, 15, 20).foreach{k =>
      val evalModel = Evaluator(k)
      val perform =
        evalModel.eval(testWithUsedItems, userPosition, itemPosition)
      println(s"[Top-$k] " + perform.toString)
    }
    // save optimized positions
    val userPosDir = jyb.concatPath(posDir, "user")
    savePosition(userPosition, userPosDir)
    val itemPosDir = jyb.concatPath(posDir, "item")
    val itemPositionRDD = sc.parallelize(itemPosition.toSeq)
    savePosition(itemPositionRDD, itemPosDir)
  }

  def savePosition(position: RDD[(Int, Point)],
                   dir: String): Boolean = {
    val sc = position.sparkContext
    jyb.deleteIfExists(sc, dir)
    position.map{
      case (i, pi) =>
        s"%d:%s".format(i, pi.mkString(","))
    }.repartition(8)
     .saveAsTextFile(dir)
    true
  }

  def formatTest(train: RDD[Metric],
                 test: RDD[jyb.Usage],
                 tempDir: String):
  RDD[(Int, Array[Int], Array[Int])] = {
    val sc = train.sparkContext
    // get warm items
    val items = train.map(_.dst)
      .distinct().collect().toSet
    val iBD = sc.broadcast(items)
    // merge usage by users in both train and test set
    val userUsedInTrain = train
      .map(p => (p.src, p.dst))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      .setName("used").persist()
    val userUsedInTest = test
      .map(p => (p.u, p.i))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      .mapValues{future => future.intersect(iBD.value)}
      .filter(_._2.nonEmpty)
      .setName("using").persist()
    // using reduce by key to combine usage
    val combined = userUsedInTrain.join(userUsedInTest)
    jyb.deleteIfExists(sc, tempDir)
    combined.map{case (u, (i1, i2)) =>
      val uStr = u.toString
      val i1Str = i1
        .map(_.toString).mkString(",")
      val i2Str = i2
        .map(_.toString).mkString(",")
      s"$uStr|$i1Str|$i2Str"
    }.repartition(test.getNumPartitions)
     .saveAsTextFile(tempDir)
    sc.textFile(tempDir).map(_.split('|'))
      .map{ps =>
        val u = ps(0).toInt
        val i1 = ps(1).split(',')
          .map(_.toInt)
        val i2 = ps(2).split(',')
          .map(_.toInt)
        (u, i1, i2)
      }
  }

  def formatToUsage(ps: Array[String]):
  jyb.Usage = {
    val u = ps(0).toInt
    val i = ps(1).toInt
    jyb.Usage(u, i)
  }

  def formatToDistance(ps: Array[String]):
  Metric = {
    val u = ps(0).toInt
    val i = ps(1).toInt
    val s = ps(2).toDouble
    val d = 1.0 - s
    Metric(u, i, d)
  }

}
