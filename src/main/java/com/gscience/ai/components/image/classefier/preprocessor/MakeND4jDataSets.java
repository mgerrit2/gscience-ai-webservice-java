package com.gscience.ai.components.image.classefier.preprocessor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Set;


/**
 * Preprocessor module designed to construct native off-heap ND4J tensor
 * datasets from standard managed Java collection structures.
 */
@Component
public class MakeND4jDataSets {

    /**
     * Managed Bean Constructor.
     * Kept public for standard Spring ApplicationContext IOC container proxy instantiation.
     */
    public MakeND4jDataSets(){
        // TODO document why this constructor is empty
    }

    /**
     * Packages aligned raw training data attributes and labels into a singular multi-dimensional DataSet.
     *
     * @param alignedData The structured images container pipeline.
     * @param bizClass    The categorical target id being targeted for binary validation.
     * @return A structured DL4J DataSet initialized off-heap.
     */
    public  DataSet makeDataSet(FeatureAndDataAligner alignedData, int bizClass) {
        INDArray alignedXData = makeDataSetTE(alignedData);
        List<Set<Integer>> labels = alignedData.getBusinessLabels();
        float[][] matrix2 = labels.stream().map(x -> (x.contains(bizClass) ? new float[]{1, 0} : new float[]{0, 1}))
                .toArray(float[][]::new);
        INDArray alignedLabs = toNDArray(matrix2);
        return new DataSet(alignedXData, alignedLabs);
    }

    /**
     * Processes individual jagged nested Integer wrappers into a contiguous matrix structure.
     *
     * @param alignedData Pre-processed feature images array wrapper.
     * @return 2D INDArray representing structural image features.
     */
    public  INDArray makeDataSetTE(FeatureAndDataAligner alignedData) {
        List<List<Integer>> imgs = alignedData.getImgVectors();
        double[][] matrix = new double[imgs.size()][];
        for (int i = 0; i < matrix.length; i++) {
            List<Integer> img = imgs.get(i);
            matrix[i] = img.stream().mapToDouble(Integer::doubleValue).toArray();
        }
        // Return clean 2D array [numImages, 4096]
        return toNDArray(matrix);
    }

    private INDArray toNDArray(float[][] matrix) {
        return Nd4j.create(matrix);
    }

    private INDArray toNDArray(double[][] matrix) {
        return Nd4j.create(matrix);
    }
}