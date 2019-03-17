package jyb.Recommender

/*
load parameters from a given xml setting files
involved settings
  1. learning rate
  2. drop-out rate
  3. clip radius
  4. nDim of latent space
  5. margin for relaxation
  6. negNum about sampling negative for training
  7. maxIterateTimes
  8. batchSize in every blocks
  9. numBlocks for users
 10. batch times in every iterate
 11. alpha for weighted push and pull loss
*/
import scala.xml.XML

case class Setting(cfgDir: String){
  require(jyb.isLocalFileExist(cfgDir), "no setting files!")

  private val xmlReader =
    XML.loadFile(cfgDir)

  private def getSetting(name: String):
  String =
    (this.xmlReader \\ name).text

  def getLearningRate: Double =
    getSetting("learningRate").toDouble

  def getDropRate: Double =
    getSetting("dropRate").toDouble

  def getClipRadius: Double =
    getSetting("clip").toDouble

  def getSpaceDim: Int =
    getSetting("nDim").toInt

  def getMargin: Double =
    getSetting("margin").toDouble

  def getNegNum: Int =
    getSetting("negNum").toInt

  def getMaxIterTimes: Int =
    getSetting("maxIterTimes").toInt

  def getBatchSize: Int =
    getSetting("batchSize").toInt

  def getBlockNum: Int =
    getSetting("blockNum").toInt

  def getBatchNumPerIter: Int =
    getSetting("batchPerIter").toInt

  def getLossWeight: Double =
    getSetting("lossWeight").toDouble
}
