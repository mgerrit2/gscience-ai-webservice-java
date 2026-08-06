package com.gscience.ai.controllers.gpu;

import com.gscience.ai.dto.gpu.GpuDiagnosticsDTO;
import com.gscience.ai.services.gpu.GpuDiagnostics;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


@Log4j2
@RequiredArgsConstructor
@RestController
@RequestMapping("/api/{version}/gpu-info")
@Tag(name= "Gifs the current GPU information")
public class GPUInfoController {

    private final GpuDiagnostics gpuDiagnostics;

    @Operation(summary = "Get the GPU ecosystem information.")
    @GetMapping(value = "", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<GpuDiagnosticsDTO> uploadDualDataStreams(
    )  {

        var responseDTO =  gpuDiagnostics.checkForMultiGPU();

        return ResponseEntity.ok(responseDTO);

    }
}
