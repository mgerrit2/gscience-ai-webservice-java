package com.gscience.ai.dto.cuda;

import lombok.Builder;
import lombok.Data;

@Data
@Builder
public class LoadTestResponse {
    private String status;
    private String backend;
    private String matrixSize;
    private long memoryAllocatedBytes;
    private long executionTimeMs;
    private String threadName;
    private String errorMessage;
}
