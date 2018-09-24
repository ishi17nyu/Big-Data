import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib._
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.evaluation.MulticlassMetrics

import org.apache.spark.rdd.RDD

object NaiveBayesAlgorithm {

  def main(args:Array[String]) : Unit = {
    val conf = new SparkConf().setAppName("NaiveBayes").setMaster("local[*]")
    val sc = new SparkContext(conf)

    val datapre = sc.textFile("../binary.csv")
    val header  = datapre.first()
    val data = datapre.filter(row => row != header)
    val parseddatapre = data.map { line =>
      val parts = line.split(',')
      LabeledPoint(parts(0).toDouble , Vectors.dense(parts(1).toInt, parts(2).toDouble, parts(3).toInt))
    }


    val splits = parseddatapre.randomSplit(Array(0.7, 0.3), seed = 17L)
    val training = splits(0)
    val test = splits(1)
    val model = NaiveBayes.train(training)
    val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
    val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

    println("the accuracy is " + accuracy)

    val metrics = new BinaryClassificationMetrics(predictionAndLabel)

    // Precision by threshold
    val precision = metrics.precisionByThreshold
    precision.foreach { case (t, p) =>
      println(s"Threshold: $t, Precision: $p")
    }

    // Recall by threshold
    val recall = metrics.recallByThreshold
    recall.foreach { case (t, r) =>
      println(s"Threshold: $t, Recall: $r")
    }

    // Precision-Recall Curve
    val PRC = metrics.pr

    val roc1 = metrics.roc

    println("the ROC is " + roc1.collect())

    val auc1 = metrics.areaUnderROC

    println("the AUC is " + auc1)

    val metrics2 = new MulticlassMetrics(predictionAndLabel)

    val confusionmatrix = metrics2.confusionMatrix
    println("**************************************************OUTPUT*******************************************")

    println("the accuracy is " + accuracy)
    println("the AUC is " + auc1)
    println("the confusionmatrix is  " + confusionmatrix)
    println("the ROC is " + roc1.collect())


    println("**************************************************OUTPUT*******************************************")

  }

}
