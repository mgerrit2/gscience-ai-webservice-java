package com.gscience.ai.services.model.cnn;

import com.gscience.ai.enumerates.PetType;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Random;

/**
 * https://ramok.tech/tag/deeplearning4j-cat-and-dog/
 */
@RequiredArgsConstructor
@Log4j2
@Service
public class VGG16Service {

    //region file paths
    @Value("${data.path.dog.versus.cat}")
    String dataPath;

    @Value("${data.path.dog.versus.cat.train}")
    String trainFolder;

    @Value("${data.path.dog.versus.cat.test}")
    String testFolder;

    @Value("${data.path.dog.versus.cat.save}")
    String saveFolder;

    @Value("${data.path.base.predefined.model}")
    String modelFolder;
    //endregion

    private ComputationGraph vgg6Graph;

    private final String savingPath = dataPath + saveFolder;

    private static final int NUM_POSSIBLE_LABELS = 2; //The target task is a binary classification problem (e.g., Cats vs. Dogs).
    private static final String FREEZE_UNTIL_LAYER = "fc2"; // only use thus layer
    private static final int EPOCH = 5;
    private static final int TRAIN_SIZE = 85;

    private static final int SAVING_INTERVAL = 100;

    private static final long seed = 12345;
    private static Random RAND_NUM_GEN = new Random(seed);
    public static final String[] ALLOWED_FORMATS = BaseImageLoader.ALLOWED_FORMATS;

    /**
     * automatically extracts classification labels for images directly from their directory structure.
     */
    public static ParentPathLabelGenerator LABEL_GENERATOR_MAKER = new ParentPathLabelGenerator();

    /**
     * balance your dataset across classes and filter out unsupported file types
     */
    public static BalancedPathFilter PATH_FILTER = new BalancedPathFilter(RAND_NUM_GEN, ALLOWED_FORMATS, LABEL_GENERATOR_MAKER);


    /**
     * Instructs DL4J to freeze all convolutional and
     * fully connected layers up to "fc2". Only the new output layer will be trained from scratch.
     */
    private static final int BATCH_SIZE = 16; //Images are fed into the network in groups of 16.

    public void train() throws IOException {

        File cacheDir = new File("./dl4j-cache");
        if (!cacheDir.exists()) {
            cacheDir.mkdirs();
        }
        System.setProperty("org.deeplearning4j.resources.cache.dir", cacheDir.getAbsolutePath());

//        long seed = 12345L; // Fixed: Defined seed variable

        String trainFolder = this.dataPath + this.trainFolder;
        String testFolder = this.dataPath + this.testFolder;

        ZooModel model = VGG16.builder().build();
        log.info("Start Downloading VGG16 model...");

        ComputationGraph preTrainedNet = (ComputationGraph) model.initPretrained(PretrainedType.IMAGENET);
        log.info(preTrainedNet.summary());


        File trainData = new File(trainFolder);
        File testData = new File(testFolder);

        //Scans the entire trainData directory (including subdirectories like /cats/ and /dogs/) and creates a list of all valid image file paths.
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, RAND_NUM_GEN);

        // training to 80% validation to 20%
        InputSplit[] sample = train.sample(PATH_FILTER, TRAIN_SIZE, 100 - TRAIN_SIZE);
        DataSetIterator trainIterator = getDataSetIterator(sample[0]);
        DataSetIterator devIterator = getDataSetIterator(sample[1]);

        // Configure fine tuning with the updated API syntax
        var fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // backpropagation algorithme
                .updater(new Adam(0.001))
                .seed(seed) // Ensures reproducibility
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .build();


        var vgg16Transfer = new TransferLearning.GraphBuilder(preTrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(FREEZE_UNTIL_LAYER)
                .removeVertexKeepConnections("predictions") // remove the output layer
                .addLayer("predictions",  // adding ne prediction layer
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // standard lose function along side softmax function
                                .nIn(4096) // Sets the number of incoming inputs
                                .nOut(NUM_POSSIBLE_LABELS) // Sets the number of outputs to your target class count
                                .weightInit(WeightInit.XAVIER) // Initializes the weights of this new layer using Xavier/Glorot initialization so training starts smoothly.
                                .activation(Activation.SOFTMAX) // Converts raw output scores into normalized class probability percentages that sum up to 1.0 (100%).
                                .build(), FREEZE_UNTIL_LAYER)
                .build();

        // Logs the training loss score every 5 minibatches
        vgg16Transfer.setListeners(new ScoreIterationListener(5));
        log.info(vgg16Transfer.summary());

        DataSetIterator testIterator = getDataSetIterator(test.sample(PATH_FILTER, 1, 0)[0]);

        int iEpoch = 0;
        int i = 0;

        while (iEpoch < EPOCH) {
            while (trainIterator.hasNext()) {
                DataSet trained = trainIterator.next();
                vgg16Transfer.fit(trained);
                if (i % SAVING_INTERVAL == 0 && i != 0) {

                    ModelSerializer.writeModel(vgg16Transfer, new File(savingPath + i + "_epoch_" + iEpoch + ".zip"), false);
                    evalOn(vgg16Transfer, devIterator, i);
                }
                i++;
            }
            trainIterator.reset();
            iEpoch++;

            evalOn(vgg16Transfer, testIterator, iEpoch);
        }

        log.info("================== end train ==================");

        //DataSetIterator testIterator = getDataSetIterator(test.sample(PATH_FILTER, 1, 0)[0]);

    }

    public static void evalOn(ComputationGraph vgg16Transfer, DataSetIterator testIterator, int iEpoch) throws IOException {
        log.info("Evaluate model at iteration " + iEpoch + " ....");

        var eval = vgg16Transfer.evaluate(testIterator);

        log.info(eval.stats());
        testIterator.reset();

    }

    /**
     * Dit is een van de belangrijkste methoden in de data-pijplijn. Deze methode zet ruwe bestanden op je schijf (InputSplit)
     * om in een gestreamde DataSetIterator die hapklare mini-batches aan afbeeldingen rechtstreeks aan het neurale netwerk kan voeren.
     * @param sample
     * @return
     * @throws IOException
     */
    public static DataSetIterator getDataSetIterator(InputSplit sample) throws IOException {

        // schaal naar 224 x 224 pixels. Dit is het exacte invoerformaat dat het VGG16-netwerk verwacht
        // Het aantal kleurkanalen (RGB).
        ImageRecordReader imageRecordReader = new ImageRecordReader(224, 224, 3, LABEL_GENERATOR_MAKER);

        //Koppelt de aangemaakte ImageRecordReader aan de specifieke
        // InputSplit (bijvoorbeeld je 85% trainingsdata of je 15% validatiedata).
        // Hij weet nu welke bestanden geladen moeten worden
        imageRecordReader.initialize(sample);

        // Wrapt de ImageRecordReader in een iterator die data in batch-formaat aanbiedt aan DL4J.
        // - BATCH_SIZE (16): De afbeeldingen worden in groepjes van 16 tegelijk ingeladen en verwerkt door de GPU/CPU.
        // - NUM_POSSIBLE_LABELS (2): Het totaal aantal klasses. Zorgt ervoor dat labels automatisch
        //   omgezet worden naar one-hot vectors (bijv. [1.0, 0.0] voor kat en [0.0, 1.0] voor hond).
        DataSetIterator iterator = new RecordReaderDataSetIterator(imageRecordReader, BATCH_SIZE, 1, NUM_POSSIBLE_LABELS);

        iterator.setPreProcessor(new VGG16ImagePreProcessor());

        return iterator;
    }

    public PetType detectCat(MultipartFile file, Double threshold) {


        /*
            Details: It automatically resizes the image to 224 × 224 pixels with 3 color channels (RGB),
            which matches the standard input dimensions required by image classification models like VGG16
         */
        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);

        try (InputStream inputStream = file.getInputStream()) {
            INDArray image = loader.asMatrix(inputStream);

            DataNormalization scaler = new VGG16ImagePreProcessor();
            scaler.transform(image);


            INDArray output = vgg6Graph.outputSingle(false, image);

            if (output.getDouble(0) > threshold) {
                return PetType.CAT;
            }else if(output.getDouble(1) > threshold){
                return PetType.DOG;
            }else{
                return PetType.NOT_KNOWN;
            }


        } catch (IOException e) {
            throw new RuntimeException(e);
        }


    }

    @PostConstruct
    public void init() throws IOException {
        vgg6Graph = loadModel();

        vgg6Graph.init();

        log.info("VGG6 is loaded");

        //log.info(vgg6Graph.summary());


    }

    private ComputationGraph loadModel() throws IOException {
        var computationGraph = ModelSerializer.restoreComputationGraph(new File(modelFolder));
        return computationGraph;
    }
}
