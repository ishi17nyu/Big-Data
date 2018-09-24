import org.apache.spark.mllib.classification.LogisticRegressionModel;
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import scala.Tuple2;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.rdd.RDD;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.log4j.LogManager;



public class RidgeRegressionFinal {

    public static final String COMMA_DELIMITER = ",(?=([^\"]*\"[^\"]*\")*[^\"]*$)";

    public static void main(String[] args) {

        LogManager.getLogger("org").setLevel(org.apache.log4j.Level.OFF);
        SparkConf sparkConf = new SparkConf().setAppName("RandomForestFinal").setMaster("local[1]");

        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        String datapath = "../binary.csv";

        JavaRDD<String> predata = sc.textFile(datapath);
        final String header = predata.first();
        JavaRDD<String> data = predata.filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String s) throws Exception {
                return !s.equalsIgnoreCase(header);
            }
        });

        JavaRDD<String>[] splits = predata.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<String> trainingData = splits[0];
        JavaRDD<String> testData = splits[1];

        //convert input to RDD label points
        JavaRDD<LabeledPoint> trainPoints = trainingData.filter(l -> !"admit".equals(l.split(COMMA_DELIMITER)[0])).map(line -> {
            String[] params = line.split(COMMA_DELIMITER);
            double label = Double.valueOf(params[0]);
            double[] vector = new double[3];
            vector[0] = Double.valueOf(params[1]);
            vector[1] = Double.valueOf(params[2]);
            vector[2] = Double.valueOf(params[3]);
            return new LabeledPoint(label, new DenseVector(vector));
        });

        JavaRDD<LabeledPoint> testPoints = testData.filter(l -> !"admit".equals(l.split(COMMA_DELIMITER)[0])).map(line -> {
            String[] params = line.split(COMMA_DELIMITER);
            double label = Double.valueOf(params[0]);
            double[] vector = new double[3];
            vector[0] = Double.valueOf(params[1]);
            vector[1] = Double.valueOf(params[2]);
            vector[2] = Double.valueOf(params[3]);
            return new LabeledPoint(label, new DenseVector(vector));
        });

        // Run training algorithm to build the model.
        LogisticRegressionModel model = new LogisticRegressionWithLBFGS()
                .run(trainPoints.rdd());

        // Clear the prediction threshold so the model will return probabilities
        model.clearThreshold();

        // Compute raw scores on the test set.
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = testPoints.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double prediction = (double)Math.round(model.predict(p.features()));
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );

        // Get evaluation metrics.
        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(predictionAndLabels.rdd());

        // Get evaluation metrics.
        MulticlassMetrics metrics2 = new MulticlassMetrics(predictionAndLabels.rdd());
        // Accuracy
        System.out.println("Accuracy = " + metrics2.accuracy());

        // Precision by threshold
        JavaRDD<Tuple2<Object, Object>> precision = metrics.precisionByThreshold().toJavaRDD();
        System.out.println("Precision by threshold: " + precision.collect());

        // Recall by threshold
        JavaRDD<?> recall = metrics.recallByThreshold().toJavaRDD();
        System.out.println("Recall by threshold: " + recall.collect());

        // F Score by threshold
        JavaRDD<?> f1Score = metrics.fMeasureByThreshold().toJavaRDD();
        System.out.println("F1 Score by threshold: " + f1Score.collect());

        JavaRDD<?> f2Score = metrics.fMeasureByThreshold(2.0).toJavaRDD();
        System.out.println("F2 Score by threshold: " + f2Score.collect());

        // Precision-recall curve
        JavaRDD<?> prc = metrics.pr().toJavaRDD();
        System.out.println("Precision-recall curve: " + prc.collect());

        // Thresholds
        JavaRDD<Double> thresholds = precision.map(t -> Double.parseDouble(t._1().toString()));

        // ROC Curve
        JavaRDD<?> roc = metrics.roc().toJavaRDD();
        System.out.println("ROC curve: " + roc.collect());

        // AUPRC
        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());

        // AUROC
        System.out.println("Area under ROC = " + metrics.areaUnderROC());

        // Save and load model

        //model.save(sc, "target/tmp/RidgeLogisticRegressionModel");
        //LogisticRegressionModel.load(sc, "target/tmp/RidgeLogisticRegressionModel");

        sc.stop();

    }
}
