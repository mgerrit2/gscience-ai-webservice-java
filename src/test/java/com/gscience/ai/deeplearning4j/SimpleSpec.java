package com.gscience.ai.deeplearning4j;

import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;


@Log4j2
public class SimpleSpec {

    @Test
    void IrisDataSet() {

        int seed = 123; // for reproducibility
        int numInputs = 4; // 4 features in Iris dataset (sepal length, sepal width, petal length, petal width)
        int numOutputs = 3; // 3 classes in Iris dataset (Iris-setosa, Iris-versicolor, Iris-virginica)
        int numHiddenNodes = 10; // Number of neurons in the hidden layer
        int numEpochs = 10000; // Number of training iterations

        log.info("Load data...");
        // Load the Iris dataset. IrisDataSetIterator provides pre-split training/testing data.
        DataSetIterator iterator = new IrisDataSetIterator(150, 150); // batch size 150, total 150 samples
        DataSet allData = iterator.next();
        allData.shuffle(seed); // Shuffle the data for better training

        // Split data into training and test sets
        // 65% for training, 35% for testing
        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        log.info("Build model...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)//Sets a random seed for the network's weight initialization
                .updater(new Nesterovs(0.005, 0.9)) // Learning rate and momentum,optimization algorithm
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU) // Hidden layer activation function
                        .weightInit(WeightInit.XAVIER) // Weight initialization
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) // Multi-class cross-entropy loss
                        .nIn(numHiddenNodes).nOut(numOutputs)
                        .activation(Activation.SOFTMAX) // Output layer activation for multi-class classification
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100)); // Print score every 100 iterations

        log.info("Train model...");
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainingData);
        }

        log.info("Evaluate model...");
        Evaluation eval = new Evaluation(numOutputs);
        eval.eval(testData.getLabels(), model.output(testData.getFeatures()));

        log.info(eval.stats());
    }
}
