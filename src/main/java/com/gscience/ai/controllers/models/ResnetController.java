package com.gscience.ai.controllers.models;
import com.gscience.ai.services.models.ResnetService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * REST controller for handling ResNet image classification requests.
 */
@RequiredArgsConstructor
@RestController
@RequestMapping("/api/v1/resnet")
public class ResnetController {

    private final ResnetService resnetService;

    /**
     * Classifies an uploaded image file and returns the top predicted labels.
     *
     * @param file the image file to be classified (JPEG, PNG, etc.)
     * @return a {@link ResponseEntity} containing a success status and top 5 predictions
     * @throws IOException if an error occurs while reading the input image stream
     */
    @Operation(
            summary = "Classify an image",
            description = "Accepts an image file upload and processes it through the ResNet model to return the top 5 predicted classifications."
    )
    @ApiResponses(value = {
            @ApiResponse(
                    responseCode = "200",
                    description = "Image classified successfully",
                    content = @Content(mediaType = MediaType.APPLICATION_JSON_VALUE)
            ),
            @ApiResponse(
                    responseCode = "400",
                    description = "Invalid image file provided or missing request parameter",
                    content = @Content
            ),
            @ApiResponse(
                    responseCode = "500",
                    description = "Internal server error during classification processing",
                    content = @Content
            )
    })
    @PostMapping("/classify")
    public ResponseEntity<?> classifyImage(@RequestParam("file") MultipartFile file) throws IOException {
        List<ResnetService.PredictionResult> predictions = resnetService.classifyImage(file.getInputStream(), 5);

        return ResponseEntity.ok(Map.of(
                "status", "success",
                "predictions", predictions
        ));
    }
}