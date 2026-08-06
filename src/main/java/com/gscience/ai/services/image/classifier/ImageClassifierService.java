package com.gscience.ai.services.image.classifier;

import com.gscience.ai.components.image.classefier.CSVImageMetadataReader;

import com.gscience.ai.components.image.classefier.ImageFeatureExtractor;
import com.gscience.ai.components.image.classefier.Networksaver;
import com.gscience.ai.components.image.classefier.evaluator.ModelEvaluation;
import com.gscience.ai.components.image.classefier.evaluator.ResultFileGenerator;
import com.gscience.ai.components.image.classefier.preprocessor.FeatureAndDataAligner;
import com.gscience.ai.components.image.classefier.preprocessor.MakeND4jDataSets;
import com.gscience.ai.components.image.classefier.trainer.CNNEpochs;
import com.gscience.ai.components.image.classefier.trainer.NetworkSaver;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

@RequiredArgsConstructor
@Log4j2
@Service
public class ImageClassifierService {


    @Value("${data.path.base}")
    String basePath;

    @Value("${data.path.csv.train}")
    String trainCSV;

    @Value("${data.path.csv.train_photo_to_biz_ids}")
    String trainPhotoToBiz;


    @Value("${data.path.csv.train.photo}")
    String trainPhoto;

    @Value("${data.path.csv.test.photo.to.biz}")
    String dataPathCsvTestPhotoToBiz;

    @Value("${data.path.csv.test.photo}")
    String dataPathCsvTestPhoto;

    @Value("${data.path.csv.result.kaggle}")
    String dataPathCsvResultKaggle;



    private final CSVImageMetadataReader csvImageMetadataReader;
    private final ImageFeatureExtractor imageFeatureExtractor;
    private final CNNEpochs cNNEpochs;
    private final ResultFileGenerator resultFileGenerator;
    private final NetworkSaver networkSaver;
    private final MakeND4jDataSets makeND4jDataSets;
    private final ModelEvaluation modelEvaluation;
    private final Networksaver networksaver;


    /**
     * train the image CNN
     * @throws IOException
     */
    public void train() throws IOException {


        log.info("basePath: {}",basePath);

        Map<String, Set<Integer>> labMap = csvImageMetadataReader.readBusinessLabels(basePath+trainCSV);

        Map<Integer, String> businessMap = csvImageMetadataReader.readBusinessToImageLabels(basePath +trainPhotoToBiz);
        List<String> businessIds = businessMap.entrySet().stream().map(Map.Entry::getValue).distinct().toList();
        List<String> imgs = imageFeatureExtractor.getImageIds(basePath+trainPhoto, businessMap, businessIds).subList(0, 100); // 20000 images
        log.info("Image ID retreival done!");

        Map<Integer, List<Integer>> dataMap = imageFeatureExtractor.processImages(imgs, 64);
        log.info("Image processing done!");

        FeatureAndDataAligner alignedData = new FeatureAndDataAligner(dataMap, businessMap, Optional.of(labMap));
        log.info("Feature extraction done!");

        /*
        for (int i = 0; i <= 8; i++) {
            String modelPath = "results/models/model" + i;
            MultiLayerNetwork model = cNNEpochs.trainModelEpochs(alignedData, i, modelPath);
            networksaver.saveModel(model, modelPath + ".zip", true);

        }
*/

        for (int i = 0; i == 0; i++) {
            String modelPath = "results/models/model" + i;
            MultiLayerNetwork model = cNNEpochs.trainModelEpochs(alignedData, i, modelPath);
            networksaver.saveModel(model, modelPath + ".zip", true);

        }

        Map<Integer, String> businessMapTE
                = csvImageMetadataReader.readBusinessToImageLabels(basePath + dataPathCsvTestPhotoToBiz);
        List<String> imgsTE = imageFeatureExtractor.getImageIds(basePath +dataPathCsvTestPhoto, businessMapTE, businessMapTE.values().stream()
                .distinct().toList()).subList(0, 100);

        Map<Integer, List<Integer>> dataMapTE = imageFeatureExtractor.processImages(imgsTE, 64); // make them 64x64
        FeatureAndDataAligner alignedDataTE = new FeatureAndDataAligner(dataMapTE, businessMapTE, Optional.empty());

        // creating csv file to submit to kaggle (scores all models)
        List<Pair<String, List<Double>>> Results = resultFileGenerator.SubmitObj(alignedDataTE, "results/models/", "model0",
                "model1", "model2", "model3", "model4",
                "model5", "model6", "model7", "model8");

        resultFileGenerator.writeSubmissionFile(dataPathCsvResultKaggle+"/"+ "kaggleSubmitFile.csv", Results, 0.50);

        // example of how to score just model
        INDArray dsTE = makeND4jDataSets.makeDataSetTE(alignedDataTE);
        MultiLayerNetwork model = networkSaver.loadNN("results/models/model0.json", "results/models/model0.bin");
        INDArray predsTE = modelEvaluation.scoreModel(model, dsTE);
        List<Pair<String, Double>> bizScoreAgg = modelEvaluation.aggImgScores2Business(predsTE, alignedDataTE);

        log.info(bizScoreAgg);

        log.info("End of evaluation");

    }
}
