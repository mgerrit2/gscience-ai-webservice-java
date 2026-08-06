package com.gscience.ai.dto.gpu;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.Min;
import lombok.Builder;
import lombok.extern.jackson.Jacksonized;

import java.io.Serial;
import java.io.Serializable;
import java.util.List;

@Builder
@Jacksonized
public record GpuDiagnosticsDTO(

        @Schema(
                description = "Check how many devices CUDA recognizes",
                example = "1052",
                minimum = "1",
                maximum = "9223372036854775807",
                accessMode = Schema.AccessMode.READ_ONLY
        )
        @Min(value = 1, message = "ID must be a positive number")
        int numDevices,

        @Schema(
                description = "List of device identifiers available for CUDA processing",
                example = "[0, 1]",
                accessMode = Schema.AccessMode.READ_ONLY
        )
        List<Integer> availableDevices,

        @Schema(
                description = "Indicates whether multi-GPU utilization is allowed and configured",
                example = "true",
                accessMode = Schema.AccessMode.READ_ONLY
        )
        boolean multiGpuAllowed,

        @Schema(
                description = "The index of the currently active GPU device",
                example = "0",
                accessMode = Schema.AccessMode.READ_ONLY
        )
        int currentDevice

) implements Serializable {
    @Serial
    private static final long serialVersionUID = 1L;
}
