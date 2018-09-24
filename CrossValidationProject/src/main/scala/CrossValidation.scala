import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession

object CrossValidation {

  val spark: SparkSession = org.apache.spark.sql.SparkSession.builder
    .master("local")
    .appName("Cross Validation")
    .getOrCreate;

  val trainingData  = spark.read
    .format("csv")
    .option("header", "true") //reading the headers
    .load("../titanic-train.csv")

  val nFolds: Int = 3
  val NumTrees: Int = 2
  val metric: String = "accuracy"

  val rf = new RandomForestClassifier()
    .setLabelCol("label")
    .setFeaturesCol("features")
    .setNumTrees(NumTrees)

  val pipeline = new Pipeline().setStages(Array(rf))

  val paramGrid = new ParamGridBuilder().build() // No parameter search

  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    // "f1" (default), "weightedPrecision", "weightedRecall", "accuracy"
    .setMetricName(metric)

  val cv = new CrossValidator()
    // ml.Pipeline with ml.classification.RandomForestClassifier
    .setEstimator(pipeline)
    // ml.evaluation.MulticlassClassificationEvaluator
    .setEvaluator(evaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(nFolds)

  val model = cv.fit(trainingData) // trainingData: DataFrame

}
