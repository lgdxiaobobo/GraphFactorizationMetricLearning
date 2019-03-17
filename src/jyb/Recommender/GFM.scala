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
    // d = 1.0 - s
    val train0 = sc.textFile(trainDir)
      .map(_.split('|'))
      .map(formatToDistance)
    val minD = train0.map(_.distance).min()
    val train = train0.map{p =>
      val d1 = (p.distance - minD) * 1.0 / (1.0 - minD)
      Metric(p.src, p.dst, d1)
    }
    println(train.map(_.distance).max())
    val test = sc.textFile(testDir)
      .map(_.split('|'))
      .map(formatToUsage)
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
      println("[Top-$k] " + perform.toString)
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
        s"%d:%s".format(i, pi.toString)
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
      .mapValues(items => (0, items))
    val userUsedInTest = test
      .map(p => (p.u, p.i))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
      .mapValues{future => future.intersect(iBD.value)}
      .filter(_._2.nonEmpty)
      .mapValues(items => (1, items))
    // using reduce by key to combine usage
    val combined = userUsedInTrain.union(userUsedInTest)
      .aggregateByKey(Array[(Int, Set[Int])]())(_ :+ _, _ ++ _)
      .filter(_._2.length > 1).map{case (u, usage) =>
        val correct = usage.sortBy(_._1).map(_._2)
        val usedInTrain = correct(0)
        val usedInTest = correct(1).diff(usedInTrain)
        (u, usedInTrain, usedInTest.toArray)
      }.filter(_._3.nonEmpty)
    jyb.deleteIfExists(sc, tempDir)
    combined.map{case (u, i1, i2) =>
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
