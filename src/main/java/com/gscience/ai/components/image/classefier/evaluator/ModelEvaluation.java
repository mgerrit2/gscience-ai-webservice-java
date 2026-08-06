package com.gscience.ai.components.image.classefier.evaluator;

import com.gscience.ai.components.image.classefier.preprocessor.FeatureAndDataAligner;
import org.apache.commons.lang3.tuple.ImmutablePair;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

@Component
public class ModelEvaluation {

    public  INDArray scoreModel(MultiLayerNetwork model, INDArray ds) {
        return model.output(ds);
    }

    /** Take model predictions from scoreModel and merge with alignedData*/
    public  List<Pair<String, Double>> aggImgScores2Business(INDArray scores,
                                                                   FeatureAndDataAligner alignedData) {

        assert(scores.size(0) == alignedData.data().size());

        ArrayList<Pair<String, Double>> result = new ArrayList<Pair<String, Double>>();

        for (String x : alignedData.getBusinessIds().stream().distinct().toList()) {
            List<String> ids = alignedData.getBusinessIds();

            // Filter down to the matching business rows
            List<Integer> matchingIndices = IntStream.range(0, ids.size())
                    .filter(i -> ids.get(i).equals(x))
                    .boxed()
                    .toList();

            // Sum the scores directly from the 2D matrix using coordinates (e, 1)
            double scoreSum = matchingIndices.stream()
                    .mapToDouble(e -> scores.getDouble(e, 1)) // <-- Fixes the bug directly!
                    .sum();

            // Calculate the business-specific mean based ONLY on the count of its own items
            double mean = matchingIndices.isEmpty() ? 0.0 : scoreSum / matchingIndices.size();

            result.add(new ImmutablePair<>(x, mean));
        }

        return result;

    }
}