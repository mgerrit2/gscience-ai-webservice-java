package com.gscience.ai.components.image.classefier.trainer;

import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.List;
import java.util.Random;

// Added missing UI/Stats storage imports
import com.gscience.ai.components.image.classefier.preprocessor.FeatureAndDataAligner;
import com.gscience.ai.components.image.classefier.preprocessor.MakeND4jDataSets;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.ui.model.stats.StatsListener;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.nd4j.evaluation.classification.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.deeplearning4j.parallelism.ParallelWrapper;

/**
 * This class, CNNEpochs, handles the end-to-end training, evaluation, and saving
 * of a 2D Convolutional Neural Network (CNN) using Eclipse Deeplearning4j (DL4J).
 */
@Log4j2
@RequiredArgsConstructor
@Component
public class CNNEpochs {

    private final MakeND4jDataSets makeND4jDataSets;

    public MultiLayerNetwork trainModelEpochs(
            FeatureAndDataAligner alignedData,
            int businessClass, String saveNN
    ) throws IOException {

        // ==================== MULTI-GPU GLOBAL CONFIG ====================
        // Allow ND4J to map execution contexts across all active GPU devices
        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);
        // =================================================================

        DataSet ds = makeND4jDataSets.makeDataSet(alignedData, businessClass);
        Logger log = LoggerFactory.getLogger(CNNEpochs.class);

        double nfeatures = (int) ds.getFeatures().getRow(0).length();
        int nlabels = (int)ds.getLabels().getRow(0).length();

        int numRows = (int) Math.sqrt(nfeatures);
        int numColumns = (int) Math.sqrt(nfeatures);

        int nChannels = 1;
        int outputNum = 2;
        int seed = 12345;
        int listenerFreq = 1;
        int nepochs = 1000;
        int nbatch = 128;

        ds.normalize();

        // ==================== CRASH FIX START ====================
        // Convert dataset to a Java list to bypass unsafe C++ off-heap splitting
        List<DataSet> listData = ds.asList();

        // Safely shuffle the aligned features and labels together using standard Java JVM memory
        Collections.shuffle(listData, new Random(seed));

        // Calculate a dynamic percentage split (e.g., 75% training, 25% testing)
        // rather than using a hardcoded row count like 75
        int totalSize = listData.size();
        int trainSize = (int) (totalSize * 0.75);
        if (trainSize == 0 && totalSize > 0) trainSize = 1; // Fallback edgecase

        List<DataSet> trainList = listData.subList(0, trainSize);
        List<DataSet> testList = listData.subList(trainSize, totalSize);

        // Safely construct iterators from JVM collections
        ListDataSetIterator<DataSet> dsiterTr = new ListDataSetIterator<>(trainList, nbatch);
        ListDataSetIterator<DataSet> dsiterTe = new ListDataSetIterator<>(testList, nbatch);
        // ==================== CRASH FIX END ======================

        MultipleEpochsIterator epochitTr = new MultipleEpochsIterator(nepochs, dsiterTr);
        MultipleEpochsIterator epochitTe = new MultipleEpochsIterator(nepochs, dsiterTe);

        // --- Network Architecture Setup ---
        ConvolutionLayer layer_0 = new ConvolutionLayer.Builder(6, 6)
                .nIn(nChannels)
                .stride(2, 2)
                .nOut(20)
                .dropOut(0.7)
                .activation(Activation.RELU)
                .build();

        SubsamplingLayer layer_1 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build();

        ConvolutionLayer layer_2 = new ConvolutionLayer.Builder(6, 6)
                .kernelSize(2, 2)
                .stride(2, 2)
                .nOut(50)
                .activation(Activation.RELU)
                .build();

        SubsamplingLayer layer_3 = new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build();

        DenseLayer layer_4 = new DenseLayer.Builder()
                .nOut(500)
                .dropOut(0.7)
                .activation(Activation.RELU)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(10)
                .build();

        OutputLayer layer_5 = new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) // <-- Changed to MCXENT
                .nOut(outputNum)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(10)
                .activation(Activation.SOFTMAX) // <-- Pairs correctly with MCXENT
                .build();

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(seed).miniBatch(true)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).l2(0.001)
                .updater(new AdaGrad(0.001)).weightInit(WeightInit.XAVIER)
                .list().layer(0, layer_0).layer(1, layer_1).layer(2, layer_2).layer(3, layer_3).layer(4, layer_4)
                .layer(5, layer_5).setInputType(InputType.convolutionalFlat(numRows, numColumns, nChannels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList(new ScoreIterationListener(listenerFreq)));

        // Print details
        Layer[] layers = model.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = (int)layers[i].numParams();

            log.info("Number of parameters in layer {}: {}", i, nParams);

            totalNumParams += nParams;
        }
        log.info("Total number of network parameters: {}" ,totalNumParams);

        // UI Server Setup
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        int listenerFrequency = 1;
        model.setListeners(new StatsListener(statsStorage, listenerFrequency));

        log.info("Train model....");
        // ==================== MULTI-GPU PARALLEL WRAPPER ====================
        // Replicates model across GPUs and coordinates data-parallel training
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                .prefetchBuffer(24)           // Number of mini-batches to prefetch
                .workers(2)                   // Set this to match your total available GPU count
                .averagingFrequency(3)        // How often gradients/weights are synchronized across devices
                .reportScoreAfterAveraging(true)
                .build();
        // ====================================================================

        log.info("Train model with multi-GPU support....");
        // Replace model.fit() with wrapper.fit()
        wrapper.fit(epochitTr);

        log.info("Evaluate model....");

        Evaluation eval = new Evaluation(outputNum);
        while (epochitTe.hasNext()) {
            DataSet testDS = epochitTe.next(nbatch);
            INDArray output = model.output(testDS.getFeatures());
            eval.eval(testDS.getLabels(), output);
        }

        if (log.isInfoEnabled()) {
            log.info(eval.stats());
        }

        if (!saveNN.isEmpty()) {
            FileUtils.write(new File(saveNN + ".json"), model.getLayerWiseConfigurations().toJson());
            DataOutputStream dos = new DataOutputStream(Files.newOutputStream(Paths.get(saveNN + ".bin")));
            Nd4j.write(model.params(), dos);
        }

        log.info("****************Example finished********************");
        return model;
    }
}