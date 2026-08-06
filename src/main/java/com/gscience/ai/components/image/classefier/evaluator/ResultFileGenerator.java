package com.gscience.ai.components.image.classefier.evaluator;

import com.gscience.ai.components.image.classefier.preprocessor.FeatureAndDataAligner;
import com.gscience.ai.components.image.classefier.preprocessor.MakeND4jDataSets;
import com.gscience.ai.components.image.classefier.trainer.NetworkSaver;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.commons.lang3.tuple.MutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Component;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.stream.Collectors;

@Log4j2
@RequiredArgsConstructor
@Component
public class ResultFileGenerator {

    private final MakeND4jDataSets makeND4jDataSets;
    private final NetworkSaver networkSaver;
    private final ModelEvaluation modelEvaluation;

    public void writeSubmissionFile(String outcsv, List<Pair<String, List<Double>>> phtoObj, double thresh) throws FileNotFoundException {

        try (PrintWriter writer = new PrintWriter(outcsv)) {

            log.info("business_ids,labels");

            for (Pair<String, List<Double>> kv : phtoObj) {

                StringBuilder sb = new StringBuilder();

                Iterator<Double> iter = kv.getValue().stream().filter(x -> x >= thresh).iterator();

                for (int idx = 0; iter.hasNext(); idx++) {
                    iter.next();
                    if (idx > 0) {
                        sb.append(' ');
                    }
                    sb.append(Integer.toString(idx));
                }

                String line = kv.getKey() + "," + sb.toString();

                log.info(line);

            }
        }

    }

    @SuppressWarnings({ "rawtypes", "unchecked" })
    public List<Pair<String, List<Double>>> SubmitObj(FeatureAndDataAligner alignedData,
                                                             String modelPath,
                                                             String model0,
                                                             String model1,
                                                             String model2,
                                                             String model3,
                                                             String model4,
                                                             String model5,
                                                             String model6,
                                                             String model7,
                                                             String model8) throws IOException {
        List<String> models = Arrays.asList(model0, model1, model2, model3, model4, model5, model6, model7, model8);

        ArrayList<Map<String, Double>> big = new ArrayList<>();

        for (String m : models) {
            INDArray ds = makeND4jDataSets.makeDataSetTE(alignedData);
            MultiLayerNetwork model = networkSaver.loadNN(modelPath + m + ".json", modelPath + m + ".bin");
            INDArray scores = modelEvaluation.scoreModel(model, ds);
            List<Pair<String, Double>> bizScores = modelEvaluation.aggImgScores2Business(scores, alignedData);
            Map<String, Double> map = bizScores.stream().collect(Collectors.toMap(Pair::getKey, Pair::getValue));
            big.add(map);
        }

        // transforming the data structure above into a List for each bizID containing a Tuple (bizid, List[Double]) where the Vector[Double] is the
        // the vector of probabilities
        List<Pair<String, List<Double>>> result = new ArrayList<>();

        Iterator<String> iter = alignedData.data().stream().map(e -> e.b).distinct().iterator();

        while (iter.hasNext()) {
            String x = iter.next();
            result.add(new MutablePair(x, big.stream().map(x2 -> x2.get(x)).toList()));
        }

        return result;
    }
}
