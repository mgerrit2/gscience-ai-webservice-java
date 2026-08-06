package com.gscience.ai.deeplearning4j.adeline;

import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

@Log4j2
class SimpleAdelineSpec {

    private final String resources_path = "src//main//resources//adeline//";

    @Test
    void trainAndSaveNetwork() {

        boolean saveUpdater = true;

        int inputSize = 2; // Number of input features
        int outputSize = 1; // ADALINE has a single output neuron
        double learningRate = 0.01;

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(learningRate))
                .list()
                .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(inputSize)
                        .nOut(outputSize)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();

        /**
         * Row 1: {0} (since 0 AND 0=0)
         * Row 2: {0} (since 0 AND 1=0)
         * Row 3: {0} (since 1 AND 0=0)
         * Row 4: {1} (since 1 AND 1=1)
         */
        INDArray input = Nd4j.create(new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        });

        INDArray labels = Nd4j.create(new double[][]{
                {0},
                {0},
                {0},
                {1}
        });

        DataSet trainingData = new DataSet(input, labels);

        // Train the model
        int nEpochs = 500;
        for (int i = 0; i < nEpochs; i++) {
            model.fit(trainingData);
        }

        // Make predictions
        INDArray testInput = Nd4j.create(new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1},
                {0.5, 0.5} // Example of an unseen input
        });

        INDArray output = model.output(testInput);
        System.out.println("Raw outputs:\n" + output);

        // Apply a threshold to get binary predictions
        INDArray predictions = output.gt(0.5);
        System.out.println("\nBinary predictions (threshold > 0.5):\n" + predictions);

        File locationToSave = new File(resources_path+ "MyAdalineModel.zip");

        try {
            ModelSerializer.writeModel(model, locationToSave, saveUpdater);
            System.out.println("Model saved successfully to " + locationToSave.getAbsolutePath());
        } catch (IOException e) {
            log.error(e);
        }

    }

    @Test
    void UseSavedNetwork() {

        File savedModelFile = new File("MyAdalineModel.zip");

        // Define the test input data (same as before)
        INDArray testInput = Nd4j.create(new double[][]{
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1},
                {0.5, 0.5} // Example of an unseen input
        });



        try {
            // Load the trained model
            // Set 'loadUpdater' to true if you saved it with updater state, false otherwise.
            boolean loadUpdater = true;
            MultiLayerNetwork restoredModel = ModelSerializer.restoreMultiLayerNetwork(savedModelFile, loadUpdater);
            System.out.println("Model loaded successfully from " + savedModelFile.getAbsolutePath());

            // --- Make predictions using the restored model ---
            INDArray rawOutput = restoredModel.output(testInput);
            System.out.println("\nRaw outputs from restored model:\n" + rawOutput);

            // Apply a threshold (e.g., 0.5 for binary classification) to get discrete predictions
            INDArray binaryPredictions = rawOutput.gt(0.5); // Greater than 0.5 will be 1, otherwise 0
            System.out.println("\nBinary predictions (threshold > 0.5) from restored model:\n" + binaryPredictions);

        } catch (IOException e) {
            System.err.println("Error loading or using the model: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
