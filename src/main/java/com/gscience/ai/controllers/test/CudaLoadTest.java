package com.gscience.ai.controllers.test;

import com.gscience.ai.dto.cuda.LoadTestResponse;
import com.gscience.ai.services.cuda.CudaLoadTestService;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@Log4j2
@RequiredArgsConstructor
@RestController
@RequestMapping("/api/{version}/cuda")
@Tag(name= "To classify images")
public class CudaLoadTest {

    private final CudaLoadTestService cudaLoadTestSv;

    @GetMapping("/matrix-multiply")
    public ResponseEntity<LoadTestResponse> triggerGpuLoad(
            @RequestParam(defaultValue = "1500") int size) {

        // Prevent accidental system locks by setting a safety limit on allocation size
        if (size > 10000) {
            return ResponseEntity.badRequest().body(
                    LoadTestResponse.builder()
                            .status("REJECTED")
                            .errorMessage("Matrix sizes above 5000x5000 are blocked to protect VRAM boundaries.")
                            .build()
            );
        }

        LoadTestResponse response = cudaLoadTestSv.runMatrixMultiplicationLoad(size);

        if ("FAILED".equals(response.getStatus())) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(response);
        }

        return ResponseEntity.ok(response);
    }

}
