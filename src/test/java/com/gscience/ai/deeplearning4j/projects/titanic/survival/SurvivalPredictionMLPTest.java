package com.gscience.ai.deeplearning4j.projects.titanic.survival;

import com.gscience.ai.deeplearning4j.projects.titanic.survival.util.SparkSessionUtil;
import com.gscience.ai.deeplearning4j.projects.titanic.survival.util.Util;
import lombok.extern.log4j.Log4j2;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.mllib.feature.StandardScalerModel;
import org.apache.spark.mllib.linalg.Vectors; // Use mllib Vectors too
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.stat.MultivariateStatisticalSummary;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.HashMap;
import java.util.Map;

/**
 * from the book "Java Deep Learning projects" chapter 1
 * GPU is not supported in this version MLP from spak
 */
@Log4j2
public class SurvivalPredictionMLPTest {

    static {
        // This forces the CUDA backend to load first
        System.setProperty("org.nd4j.backend.priority", "100");
    }

    @BeforeAll
    static void checkBackend() {
        log.info("Backend Name: " + Nd4j.getBackend().getClass().getName());
        log.info("Devices: " + Nd4j.getAffinityManager().getNumberOfDevices());
    }

    @Test
    void survivalPrediction() throws Exception {
        // setup spark
        SparkSession spark = SparkSessionUtil.getInstance();

        // 2. DATA LOADING: Retrieve the raw Titanic training data
        Dataset<Row> trainingDF = Util.getTrainingDF();

        // 3. STATISTICAL PREPARATION: Use pre-calculated summary stats for normalization
        // from fare and age
        MultivariateStatisticalSummary summary = Util.summary;
        double meanFare = summary.mean().apply(0);
        double meanAge = summary.mean().apply(1);

        // Create Vectors for standard deviation and mean to be used by the Scaler
        Vector stddev = Vectors.dense(Math.sqrt(summary.variance().apply(0)), Math.sqrt(summary.variance().apply(1)));
        Vector mean = Vectors.dense(summary.mean().apply(0), summary.mean().apply(1));

        // Initialize the Scaler to shift data to a standard range (Z-score normalization)
        StandardScalerModel scaler = new StandardScalerModel(stddev, mean);

        // 4. ENCODING: Define encoders for converting RDD objects back into DataFrames
        Encoder<Integer> integerEncoder = Encoders.INT();
        Encoder<Double> doubleEncoder = Encoders.DOUBLE();
        Encoders.BINARY();
        Encoder<Vector> vectorEncoder = Encoders.kryo(Vector.class);
        Encoders.tuple(integerEncoder, vectorEncoder);
        Encoders.tuple(doubleEncoder, vectorEncoder);

        // 5. FEATURE ENGINEERING: Transform raw Rows into Normalized Feature Vectors
        JavaRDD<Util.VectorPair> scaledRDD = trainingDF.toJavaRDD().map(row -> {
            Util.VectorPair vectorPair = new Util.VectorPair();

            // Set the Target Label (0 or 1)
            vectorPair.setLable(Double.valueOf(row.<Integer>getAs("Survived")));

            // Create a single Vector containing all features, scaled via the StandardScaler
            vectorPair.setFeatures(Util.getScaledVector(
                    row.<Double>getAs("Fare"),
                    row.<Double>getAs("Age"),
                    row.<Integer>getAs("Pclass"),
                    row.<Integer>getAs("Sex"),
                    row.isNullAt(7) ? 0d : row.<Integer>getAs("Embarked"),
                    scaler));

            return vectorPair;
        });

        // Convert RDD back to DataFrame and ensure compatibility with Spark ML
        System.out.println("RDD to dataframe");
        Dataset<Row> scaledDF = spark.createDataFrame(scaledRDD, Util.VectorPair.class);
        scaledDF.show();
        Dataset<Row> scaledData2 = MLUtils.convertVectorColumnsToML(scaledDF);

        // Rename columns to "features" and "label" as required by Spark ML algorithms
        Dataset<Row> data = scaledData2.toDF("features", "label");

        // 6. DATA SPLITTING: 80% for training the model, 20% for testing accuracy
        Dataset<Row>[] datasets = data.randomSplit(new double[]{0.80, 0.20}, 12345L);
        Dataset<Row> trainingData = datasets[0];
        Dataset<Row> validationData = datasets[1];

        // 7. MODEL ARCHITECTURE: Input(10) -> Hidden(16) -> Hidden(32) -> Output(2)
        //int[] layers = new int[] {10, 16, 32, 2};
        int[] layers = new int[] {10, 64, 32,16, 2}; // diamont shape is better

        // 8. TRAINING: Configure the Multilayer Perceptron (Neural Network)
        // It uses the Sigmoid activation function for hidden layers and Softmax for the output layer.
        MultilayerPerceptronClassifier mlp = new MultilayerPerceptronClassifier()
                .setLayers(layers)
                .setBlockSize(128)   // Stack size for optimization
                .setSeed(1234L)      // For reproducibility
                 .setTol(1E-8)        // Convergence tolerance
         .setMaxIter(1000);   // Max training 1000, bigger means slower but more accurate;


        // Run the training algorithm. This produce the most calculation
        // uses:
        //   - The Optimization Engine: L-BFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
        //
        MultilayerPerceptronClassificationModel model = mlp.fit(trainingData);

        // 9. VALIDATION: Run predictions on the 20% hold-out data
        System.out.println("show predictions");
        Dataset<Row> predictions = model.transform(validationData);
        predictions.show();

        // 10. EVALUATION: Calculate performance metrics
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label").setPredictionCol("prediction");

        MulticlassClassificationEvaluator evaluator1 = evaluator.setMetricName("accuracy");
        MulticlassClassificationEvaluator evaluator2 = evaluator.setMetricName("weightedPrecision");
        MulticlassClassificationEvaluator evaluator3 = evaluator.setMetricName("weightedRecall");
        MulticlassClassificationEvaluator evaluator4 = evaluator.setMetricName("f1");

        // compute the classification accuracy, precision, recall, f1 measure and error on test data.
        double accuracy = evaluator1.evaluate(predictions);
        double precision = evaluator2.evaluate(predictions);
        double recall = evaluator3.evaluate(predictions);
        double f1 = evaluator4.evaluate(predictions);

        // Output performance to console
        System.out.println("Accuracy = " + accuracy);
        System.out.println("Precision = " + precision);
        System.out.println("Recall = " + recall);
        System.out.println("F1 = " + f1);
        System.out.println("Test Error = " + (1 - accuracy));

        // 11. INFERENCE: Load the actual "Test" data (where survived is unknown)
        Dataset<Row> testDF = Util.getTestDF();
        testDF.show();

        // Fill missing Age/Fare values using the training set means (prevents data leakage)
        Map<String, Object> m = new HashMap<String, Object>();
        m.put("Age", meanAge);
        m.put("Fare", meanFare);
        Dataset<Row> testDF2 = testDF.na().fill(m);
        testDF2.show();

        // Prepare Test data vectors
        JavaRDD<Util.VectorPair> testRDD = testDF2.javaRDD().map(row -> {
            Util.VectorPair vectorPair = new Util.VectorPair();
            vectorPair.setLable(row.<Integer>getAs("PassengerId"));
            vectorPair.setFeatures(Util.getScaledVector(
                    row.<Double>getAs("Fare"),
                    row.<Double>getAs("Age"),
                    row.<Integer>getAs("Pclass"),
                    row.<Integer>getAs("Sex"),
                    row.<Integer>getAs("Embarked"),
                    scaler));
            return vectorPair;
        });

        // Final prediction on test set
        Dataset<Row> scaledTestDF = spark.createDataFrame(testRDD, Util.VectorPair.class);

        Dataset<Row> finalTestDF = MLUtils.convertVectorColumnsToML(scaledTestDF).toDF("features", "PassengerId");
        trainingData.show();
        finalTestDF.show();

        //w(); 12. EXPORT: Save results to a CSV file for submission
        Dataset<Row> resultDF = model.transform(finalTestDF).select("PassengerId", "prediction");
        resultDF.show();
        resultDF.write()
                .format("com.databricks.spark.csv")
                .option("header", true)
                .mode("overwrite") // This automatically deletes the old folder/files
                .save("result/resultTitanicSurvivor.csv");

    }
}
