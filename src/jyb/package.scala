/*
here we define basic functions for all works
including:
  spark related functions
  spark-sql related functions
  basic process about spark IO
*/

import org.apache.hadoop.fs
import fs.{FileSystem, Path}

import org.apache.spark
import spark.SparkContext
import spark.rdd.RDD
import spark.sql
import sql.SparkSession

import java.nio.file
import file.{Files, Paths}

package object jyb {

  case class Usage(u: Int, i: Int){
    override def toString: String =
      s"$u|$i"
  }

  def buildUsage(ps: Array[String]):
  Usage = {
    val u = ps(0).toInt
    val i = ps(1).toInt
    Usage(u, i)
  }

  def saveUsage(data: RDD[Usage],
                partNum: Int,
                dir: String):
  Boolean = {
    val sc = data.sparkContext
    deleteIfExists(sc, dir)
    data.map(_.toString)
      .repartition(partNum)
      .saveAsTextFile(dir)
    true
  }

  def getSS(name: String):
  SparkSession = {
    SparkSession.builder()
      .appName(name)
      .getOrCreate()
  }

  def deleteIfExists(sc: SparkContext,
                     dir: String):
  Boolean = {
    val hfs =
      FileSystem.get(sc.hadoopConfiguration)
    val hPath = new Path(dir)
    if (isDirExists(sc, dir)){
      println(s"Delete existed directory $dir")
      hfs.delete(hPath, true)
    }
    true
  }

  def deleteIfExists(sc: SparkContext,
                     dir: String, quite: Boolean):
  Boolean = {
    val hfs =
      FileSystem.get(sc.hadoopConfiguration)
    val hPath = new Path(dir)
    if (isDirExists(sc, dir)){
      if (!quite)
        println(s"Delete existed directory $dir")
      hfs.delete(hPath, true)
    }
    true
  }

  def isDirExists(sc: SparkContext,
                  dir: String):
  Boolean = {
    val hfs =
      FileSystem.get(sc.hadoopConfiguration)
    val hPath =
      new Path(dir)
    hfs.exists(hPath)
  }

  def isLocalFileExist(dir: String):
  Boolean = {
    val lPath = Paths.get(dir)
    Files.exists(lPath)
  }

  def concatPath(baseDir: String,
                 addName: String):
  String = {
    Paths.get(baseDir, addName)
      .toString
  }

}
