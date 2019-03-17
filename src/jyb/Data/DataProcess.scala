package jyb.Data

/*
data process for real-world dataset
which storing as t|u|a
where u for user, and a for app
here we only consider the interaction as <u, a>
the temporal (t) will be considered in the future
*/
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import scala.util.Random

object DataProcess {

  def main(args: Array[String]): Unit = {
    val dataDir = args(0)
    println(s"load interactions from $dataDir")

    val ss = jyb.getSS("DataProcess")
    val sc = ss.sparkContext
    require(jyb.isDirExists(sc, dataDir), "there is no input data")

    val gamma = sc.textFile(dataDir)
      .map(_.split('|').tail)
      .map(jyb.buildUsage)
    /*
    in order to avoiding over-fitting
    we filter both users and apps
      with less than 10 interactions
    */
    learnInfo(gamma)
    val gammaU = gamma.map(p => (p.u, p.i))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
    val filterByUser = gammaU.filter(_._2.size >= 10)
      .flatMap{case (u, pu) => pu.map(i => jyb.Usage(u, i))}
    val gammaA = filterByUser.map(p => (p.i, p.u))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
    val filterByApp = gammaA.filter(_._2.size >= 10)
      .flatMap{case (i, pi) => pi.map(u => jyb.Usage(u, i))}
    learnInfo(filterByApp)

    val dstDir = args(1)
    jyb.saveUsage(renumbered(ss, filterByApp),
      gamma.getNumPartitions, dstDir)

    /*
    strained training dataset into two parts
    one for training and the other for validation
    splitting weight is 4:1
    */
    val cvDir = args(2)
    val trainDir = jyb.concatPath(cvDir, "train")
    val validDir = jyb.concatPath(cvDir, "valid")
    val interaction = sc.textFile(dstDir)
      .map(_.split('|')).map(jyb.buildUsage)
    val nPart = interaction.getNumPartitions
    val seed = System.nanoTime()
    val rng = new Random(seed)
    val (train, valid) =
      strainedSplit(interaction, rng.nextLong(), 0.8)
    println("info about train-set")
    learnInfo(train)
    println("info about valid-set")
    learnInfo(valid)
    jyb.saveUsage(train, nPart, trainDir)
    jyb.saveUsage(valid, nPart, validDir)
  }

  def strainedSplit(data: RDD[jyb.Usage],
                    seed: Long, wTrain: Double):
  (RDD[jyb.Usage], RDD[jyb.Usage]) = {
    val rng = new Random(seed)
    val mergeByUser = data.map(p => (p.u, p.i))
      .aggregateByKey(Set[Int]())(_ + _, _ ++ _)
    val splitted = mergeByUser
      .filter(_._2.size >= 10)
      .mapValues{ pu =>
        val items = rng.shuffle(pu.toSeq)
        val sz = (pu.size * wTrain).toInt
        (items.take(sz), items.drop(sz))
      }
    val part1 = splitted.mapValues(_._1)
      .flatMap{case (u, is) =>
        is.map(i => jyb.Usage(u, i))
      }
    val part2 = splitted.mapValues(_._2)
      .flatMap{case (u, is) =>
        is.map(i => jyb.Usage(u, i))
      }
    (part1, part2)
  }

  def renumbered(ss: SparkSession,
                 interaction: RDD[jyb.Usage]):
  RDD[jyb.Usage] = {
    import ss.implicits._
    val df0 = interaction
      .toDF("u", "i")
    val userIdx = interaction.map(_.u)
      .distinct().zipWithIndex()
      .toDF("u", "idx")
    val itemIdx = interaction.map(_.i)
      .distinct().zipWithIndex()
      .toDF("i", "idy")
    val df1 = df0.join(userIdx, "u")
      .join(itemIdx, "i")
      .select("idx", "idy")
      .toDF("u", "i")
    df1.as[(Long, Long)].rdd
      .map{
        case (uL, iL) =>
          jyb.Usage(uL.toInt, iL.toInt)
      }
  }

  def learnInfo(interaction: RDD[jyb.Usage]):
  Unit = {
    val nUsers = interaction.map(_.u)
      .distinct().count()
    val nItems = interaction.map(_.i)
      .distinct().count()
    val nInteractions = interaction.count()
    println(s"[INFO] $nUsers,$nItems,$nInteractions")
  }

}
