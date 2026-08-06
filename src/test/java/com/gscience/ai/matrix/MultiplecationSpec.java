package com.gscience.ai.matrix;

import lombok.extern.log4j.Log4j2;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/**
 * {1, 5} is the row
 */
@Log4j2
public class MultiplecationSpec {


    /**
     * sample transpose matrix
     */
    @Test
    void transpose() {

        INDArray A = Nd4j.create(new double[][]{{7, 9,8,1},{9,3,-5,0}});
        log.info("Original Matrix A (Shape: {}):", A);

        var result = A.transpose();
        log.info("Transposed Matrix (Shape: {}):", result);

        // Define the expected transposed matrix for verification
        INDArray expectedResult = Nd4j.create(new double[][]{{7, 9}, {9, 3}, {8, -5}, {1, 0}});

        // Assert that the transposed matrix is equal to the expected result
        Assertions.assertEquals(expectedResult, result, "The transposed matrix should match the expected result.");

    }

    @Test
    void add() {
        INDArray A = Nd4j.create(new double[][]{{2, 1,0},{2,6,-1}});
        INDArray B = Nd4j.create(new double[][]{{8, 0,5},{0,1,7}});

        var result = A.add(B);


        log.info("Added (Shape: {}):", result);

        // Define the expected result of the matrix addition
        INDArray expectedResult = Nd4j.create(new double[][]{{10, 1, 5}, {2, 7, 6}});

        // Assert that the added matrix is equal to the expected result
        Assertions.assertEquals(expectedResult, result, "The added matrix should match the expected result.");

    }

    /*
    @Test
    void inverse() {
        // A square matrix is required to find the inverse
        INDArray A = Nd4j.create(new double[][]{{2, 1}, {4, 3}});
        log.info("Original Matrix A (Shape: {}): {}", A.shape(), A);

        // Perform the inverse operation
        INDArray result = A.inverse();
        log.info("Inverted Matrix (Shape: {}): {}", result.shape(), result);

        // The product of a matrix and its inverse should be the identity matrix
        INDArray product = A.mmul(result);
        log.info("Product of A and its inverse (Shape: {}): {}", product.shape(), product);

        // Define the expected identity matrix for verification
        INDArray identityMatrix = Nd4j.eye(2);

        // Assert that the product is close to the identity matrix (using a small tolerance for floating point errors)
        Assertions.assertTrue(product.equalsWithEps(identityMatrix, 1e-9), "The product of a matrix and its inverse should be the identity matrix.");
    }
    
     */

}
