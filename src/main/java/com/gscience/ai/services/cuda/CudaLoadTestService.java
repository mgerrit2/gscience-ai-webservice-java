package com.gscience.ai.services.cuda;

import com.gscience.ai.dto.cuda.LoadTestResponse;
import lombok.extern.log4j.Log4j2;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Service;

@Log4j2
@Service
public class CudaLoadTestService {

    public LoadTestResponse runMatrixMultiplicationLoad(int size) {
        long startTime = System.currentTimeMillis();
        String currentThread = Thread.currentThread().getName();

        // Calculate raw off-heap footprint: 2 matrices * (size * size) * 4 bytes (Float)
        long estimatedMemoryBytes = (long) size * size * 4 * 2;

        log.info("[{}] Initializing {}x{} matrix transformation on GPU...", currentThread, size, size);

        // Allocate directly on device VRAM using try-with-resources to clear pointers immediately
        try (INDArray matrixA = Nd4j.rand(size, size);
             INDArray matrixB = Nd4j.rand(size, size)) {

            try (INDArray result = matrixA.mmul(matrixB)) {

                // Force CUDA pipeline execution queue push
                Nd4j.getExecutioner().commit();

                long duration = System.currentTimeMillis() - startTime;
                log.info("[{}] Completed GEMM operations in {}ms", currentThread, duration);

                return LoadTestResponse.builder()
                        .status("SUCCESS")
                        .backend(Nd4j.getBackend().toString())
                        .matrixSize(size + "x" + size)
                        .memoryAllocatedBytes(estimatedMemoryBytes)
                        .executionTimeMs(duration)
                        .threadName(currentThread)
                        .build();
            }
        } catch (Throwable t) {
            log.error("CRITICAL: CUDA Runtime Exception during load processing: ", t);

            // REMOVED Pointer.deallocate();
            // The try-with-resources auto-closes matrixA and matrixB buffers seamlessly,
            // even when a runtime matrix execution panic triggers.

            return LoadTestResponse.builder()
                    .status("FAILED")
                    .backend(Nd4j.getBackend() != null ? Nd4j.getBackend().toString() : "UNKNOWN")
                    .matrixSize(size + "x" + size)
                    .memoryAllocatedBytes(0)
                    .executionTimeMs(System.currentTimeMillis() - startTime)
                    .threadName(currentThread)
                    .errorMessage(t.getMessage())
                    .build();
        }
    }

}
