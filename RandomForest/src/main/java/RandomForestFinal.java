import org.apache.log4j.LogManager;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.api.java.function.Function;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;

public class RandomForestFinal {

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

        System.out.println("total train data:" + trainPoints.count());
        System.out.println("total test:" + testPoints.count());

        // Train a RandomForest model
        // Empty categoricalFeaturesInfo indicates all features are continuous
        Integer numClasses = 2;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        Integer numTrees = 5; // Use more in practice.
        String featureSubsetStrategy = "auto"; // Let the algorithm choose.
        String impurity = "gini";
        Integer maxDepth = 4;
        Integer maxBins = 32;
        Long seed = 17L;

        RandomForestModel model = RandomForest.trainClassifier(trainPoints, numClasses,
                categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
                Math.toIntExact(seed));

        // Evaluate model on test instances and compute test error
        JavaPairRDD<Object, Object> predictionAndLabel =
                testPoints.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        double testErr =
                predictionAndLabel.filter(pl -> !(pl._1()).equals(pl._2())).count() / (double) testData.count();
        System.out.println("Accuracy: " + testErr);
        //System.out.println("Learned classification forest model:\n" + model.toDebugString());

        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = testPoints.map(
                new Function<LabeledPoint, Tuple2<Object, Object>>() {
                    public Tuple2<Object, Object> call(LabeledPoint p) {
                        Double prediction = model.predict(p.features());
                        return new Tuple2<Object, Object>(prediction, p.label());
                    }
                }
        );





        BinaryClassificationMetrics metrics =
                new BinaryClassificationMetrics(predictionAndLabel.rdd());

        MulticlassMetrics metrics2 = new MulticlassMetrics(predictionAndLabels.rdd());
        double precision = metrics2.precision(metrics2.labels()[0]);




        long truePositive = predictionAndLabel.filter(pl -> (pl._1()).equals(pl._2()) && pl._1().equals(1d)).count();
        System.out.println("True Positive: " + truePositive);

        long trueNegative = predictionAndLabel.filter(pl -> (pl._1()).equals(pl._2()) && pl._1().equals(0d)).count();
        System.out.println("True Negative: " + trueNegative);

        long falsePositive = predictionAndLabel.filter(pl -> !(pl._1()).equals(pl._2()) && pl._2().equals(0d)).count();
        System.out.println("False Positive: " + falsePositive);

        long falseNegative = predictionAndLabel.filter(pl -> !(pl._1()).equals(pl._2()) && pl._2().equals(1d)).count();
        System.out.println("False Negative:" + falseNegative);

        System.out.println("\nConfusion Matrix:");
        System.out.println("n: " + testData.count() + "   0" + "   1");
        System.out.println("0: " + "   " + trueNegative + "   " + falseNegative);
        System.out.println("1: " + "   " + falsePositive + "   " + truePositive + "\n");

        //Precision
        System.out.println("Precision = " + precision);


        // AUPRC
        System.out.println("Area under precision-recall curve = " + metrics.areaUnderPR());

        // AUROC
        System.out.println("Area under ROC = " + metrics.areaUnderROC());

        sc.stop();
    }
}
