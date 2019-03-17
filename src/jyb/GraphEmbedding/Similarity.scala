package jyb.GraphEmbedding

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.{functions => fn}

/*
 here we do the following process for further modelling
 1. load training data
 2. learning user-item similarity based given strategy
 3. storing Similarity[u, i, sui] for relevant user-item pairs
*/

object Similarity {

  val csver = "com.databricks.spark.csv"

  def main(args: Array[String]): Unit = {
    val trainDir = args(0)
    val embedDir = args(1)
    val method = args(2)
    val alpha = method match {
      case "jaccard" => 0.0
      case _ => args(3).toDouble
    }
    val ss = jyb.getSS("biGraph")
    val sc = ss.sparkContext
    require(jyb.isDirExists(sc, trainDir), s"$trainDir not exists")
    val headers = Array("u", "i")
    val df0 = loadDataFrame(ss, trainDir, headers)
      .as("X0").persist()
    val df1 = method match {
      case "jaccard" => jaccardCoeff(df0)
      case "aa" => aaCoeff(df0, alpha)
      case "tfidf" => tfidfCoeff(df0)
    }
    val similarityDir = jyb.concatPath(embedDir, method)
    jyb.deleteIfExists(sc, similarityDir)
    saveDataFrame(df1, similarityDir)
  }

  def saveDataFrame(df: DataFrame,
                    dir: String):
  Boolean = {
    df.write.format(csver)
      .option("inferSchema", "false")
      .option("header", "false")
      .option("sep", "|").save(dir)
    true
  }

  def tfidfCoeff(df0: DataFrame):
  DataFrame = {
    val ss = df0.sparkSession
    import ss.implicits._

    val N = df0.select("i")
      .distinct().count()

    val idf = fn.udf((n: Int) => {
      math.log(N * 1.0 / n)
    })
    val df = fn.udf((n: Int) => {
      1.0 / n
    })
    val mul = fn.udf((x: Double, y: Double) => {
      x * y
    })

    val userWeight = df0.groupBy("u")
      .agg(fn.countDistinct("i").as("nu"))
      .withColumn("wu", idf(fn.col("nu")))
      .select(df0("u"), fn.col("wu"))
      .toDF("u", "wu")
      .as("WU").persist()
    val itemWeight = df0.groupBy("i")
      .agg(fn.countDistinct("u").as("ni"))
      .withColumn("wi", df(fn.col("ni")))
      .select(df0("i"), fn.col("wi"))
      .toDF("i", "wi")
      .as("WI").persist()
    val userItemScore = df0.join(userWeight, "u")
      .join(itemWeight, "i")
      .withColumn("sui", mul(userWeight("wu"), itemWeight("wi")))
      .select(df0("u"), df0("i"), fn.col("sui"))
      .toDF("u", "i", "sui")
      .as("SUI").persist()
    userItemScore.count()
    itemWeight.unpersist(); userWeight.unpersist()

    val maxScore = userItemScore.select("sui")
      .as[Double].rdd.max()
    val scaled = fn.udf((x: Double) => {
      x / maxScore
    })
    val nUserItemScore = userItemScore
      .withColumn("nSui", scaled(fn.col("sui")))
      .select("u", "i", "nSui")
      .toDF("u", "i", "sui")
    nUserItemScore.count()
    userItemScore.unpersist()
    nUserItemScore
  }

  def aaCoeff(df0: DataFrame,
              alpha: Double):
  DataFrame = {
    val ss = df0.sparkSession
    import ss.implicits._

    val aaWeight = fn.udf(
      (n: Int) =>
        1.0 / (alpha + math.log(n))
    )
    val normalise = fn.udf(
      (sij: Double, si: Double, sj: Double) =>
        if (si * sj == 0)
          0.0
        else
          sij / math.sqrt(si * sj)
    )
    val userWeights = df0.groupBy("u")
      .agg(fn.countDistinct("i").as("nu"))
      .withColumn("wu", aaWeight(fn.col("nu")))
      .select("u", "wu")
      .as("WU").persist()
    // get cn => <i, j, u>
    val t1 = df0.join(userWeights, "u")
      .select(df0("u"), df0("i"), userWeights("wu"))
      .toDF("u", "i", "wu")
      .as("T1").persist()
    t1.count()
    userWeights.unpersist()
    val t2 = t1.toDF("u", "j", "wu")
      .as("T2").persist()
    val n1 = t1.groupBy("i")
      .agg(fn.sum("wu").as("si"))
      .select("i", "si")
      .as("Ni").persist()
    val itemSim = t1.join(t2, "u")
      .groupBy(t1("i"), t2("j"))
      .agg(fn.sum(t1("wu")).as("sij"))
      .select("T1.i", "T2.j", "sij")
      .as("SIJ").persist()
    n1.count(); itemSim.count()
    t2.unpersist(); t1.unpersist()
    val n2 = n1.toDF("j", "sj")
      .as("Nj").persist()
    val nItemSim = itemSim.join(n1, "i")
      .join(n2, "j")
      .withColumn("nSij", normalise(itemSim("sij"), n1("si"), n2("sj")))
      .select(itemSim("i"), itemSim("j"), fn.col("nSij"))
      .toDF("i", "j", "sij")
      .as("NSIJ").persist()
    nItemSim.count()
    itemSim.unpersist()
    n2.unpersist(); n1.unpersist()
    // <u, i> + <u, j> => <u, i, j>
    // <u, i, j> + <i, j, sij> => <u, i, sij>
    val df2 = df0.withColumnRenamed("i", "j")
    val refer = df0.join(df2, "u")
      .select(df0("u"), df0("i"), df2("j"))
      .toDF("u", "i", "j")
    val userItemScore = refer.join(nItemSim,
      refer("i") === nItemSim("i") && refer("j") === nItemSim("j")
    ).groupBy(refer("u"), refer("i"))
     .agg(fn.mean(nItemSim("sij")).as("sui"))
     .select(refer("u"), refer("i"), fn.col("sui"))
     .toDF("u", "i", "sui")
    userItemScore.count()
    nItemSim.unpersist()
    userItemScore
  }

  def jaccardCoeff(df0: DataFrame):
  DataFrame = {
    val ss = df0.sparkSession
    import ss.implicits._

    // <i, j, cn_ij>
    val df1 = df0.toDF("u", "j")
      .as("X1").persist()
    val commonNeighbors = df0.join(df1, "u")
      .groupBy(df0("i"), df1("j"))
      .agg(fn.countDistinct("u").as("cnij"))
      .as("CN").persist()
    commonNeighbors.count()
    df1.unpersist()
    // <i, ni>
    val itemNeighbors1 = df0.groupBy("i")
      .agg(fn.countDistinct("u").as("ni"))
      .select("i", "ni")
      .as("T1").persist()
    val itemNeighbors2 = itemNeighbors1.toDF("j", "nj")
      .as("T2").persist()
    val jaccardScore = fn.udf(
      (cn: Int, n1: Int, n2: Int) =>
        if (n1 + n2 == 0)
          0.0
        else
          cn * 1.0 / (n1 + n2 - cn)
    )
    val itemSim = commonNeighbors.join(itemNeighbors1, "i")
      .join(itemNeighbors2, "j")
      .withColumn("sij",
        jaccardScore(
          fn.col("CN.cnij"),
          fn.col("T1.ni"),
          fn.col("T2.nj")
        )
      ).select("CN.i", "CN.j", "sij")
      .toDF("i", "j", "sij")
      .as("SIJ").persist()
    itemSim.count()
    itemNeighbors2.unpersist(); itemNeighbors1.unpersist()
    commonNeighbors.unpersist()
    // <u, i> + <u, j> => <u, i, j>
    // <u, i, j> + <i, j, sij> => <u, i, sij>
    val df2 = df0.withColumnRenamed("i", "j")
    val refer = df0.join(df2, "u")
      .select(df0("u"), df0("i"), df2("j"))
      .toDF("u", "i", "j")
    val userItemScore = refer.join(itemSim, refer("i") === itemSim("i") && refer("j") === itemSim("j"))
      .groupBy(refer("u"), refer("i"))
      .agg(fn.mean(itemSim("sij")).as("sui"))
      .select(refer("u"), refer("i"), fn.col("sui"))
      .toDF("u", "i", "sui")
    userItemScore.count()
    itemSim.unpersist()
    userItemScore
  }

  def loadDataFrame(ss: SparkSession,
                    dir: String,
                    headers: Array[String]):
  DataFrame = {
    ss.read.format(csver)
      .option("inferSchema", "false")
      .option("header", "false")
      .option("sep", "|").load(dir)
      .toDF(headers:_*)
  }

}
