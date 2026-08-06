package com.gscience.ai.controllers.models;

import com.gscience.ai.components.image.classefier.Networksaver;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;


@Log4j2
@RestController
@RequestMapping("/api/{version}/models")
@RequiredArgsConstructor
@Tag(name = "REST controller responsible for handling model download endpoints,including single model file retrieval and group bundle archives.")
public class ModelController {

    /**
     * Base directory path where trained models are stored, injected from application properties.
     */
    @Value("${data.path.base.model.result}")
    String modelDir;

    /**
     * Service component handling underlying file management and model storage operations.
     */
    private final Networksaver networksaver;

    /**
     * Downloads an individual model file (.zip) by its unique identifier or name.
     *
     * @param modelName The name of the model to download (appends .zip automatically if missing).
     * @return ResponseEntity containing the file resource as an octet stream, or 404/500 status on failure.
     */
    @Operation(
            summary = "Downloads an individual model file",
            description = "Downloads an individual model file (.zip) by its unique identifier or name.",
            responses = {
                    @ApiResponse(
                            responseCode = "200",
                            description = "Successfully retrieved and downloaded the model file.",
                            content = @Content(mediaType = MediaType.APPLICATION_OCTET_STREAM_VALUE, schema = @Schema(type = "string", format = "binary"))
                    ),
                    @ApiResponse(
                            responseCode = "404",
                            description = "Model file not found."
                    ),
                    @ApiResponse(
                            responseCode = "500",
                            description = "Internal Server Error"
                    )
            })
    @GetMapping("/download/{modelName}")
    public ResponseEntity<Resource> downloadModelZip(
            @Parameter(
                    description = "The name of the model to download (appends .zip automatically if missing).",
                    required = true
            )
            @PathVariable
            String modelName
    ) {
        try {
            String fileName = modelName.endsWith(".zip") ? modelName : modelName + ".zip";
            Path filePath = Paths.get(modelDir).resolve(fileName).normalize();
            Resource resource = new UrlResource(filePath.toUri());

            if (resource.exists() && resource.isReadable()) {
                return ResponseEntity.ok()
                        .contentType(MediaType.APPLICATION_OCTET_STREAM)
                        .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + resource.getFilename() + "\"")
                        .body(resource);
            } else {
                return ResponseEntity.notFound().build();
            }
        } catch (Exception e) {
            log.error("Failed to download model file: {}", modelName, e);
            return ResponseEntity.internalServerError().build();
        }
    }

    /**
     * Bundles and downloads multiple models sharing a base name pattern into a single zip archive.
     *
     * @param baseName The base prefix pattern matching target models (e.g., "model").
     * @return ResponseEntity containing the bundled zip file resource, or 404/500 status on failure.
     */
    @Operation(
            summary = "Downloads a group bundle of models",
            description = "Bundles and downloads multiple models matching a common base name pattern into a single zip file.",
            responses = {
                    @ApiResponse(
                            responseCode = "200",
                            description = "Successfully bundled and downloaded the model group archive.",
                            content = @Content(mediaType = MediaType.APPLICATION_OCTET_STREAM_VALUE, schema = @Schema(type = "string", format = "binary"))
                    ),
                    @ApiResponse(
                            responseCode = "404",
                            description = "No models found matching the specified base name."
                    ),
                    @ApiResponse(
                            responseCode = "500",
                            description = "Internal Server Error"
                    )
            })
    @GetMapping("/download/group/{baseName}")
    public ResponseEntity<Resource> downloadModelGroupZip(
            @Parameter(
                    description = "The base prefix pattern matching target models (e.g., 'model').",
                    required = true
            )
            @PathVariable String baseName
    ) {
        try {
            String zipName = baseName + "_group_bundle.zip";

            // Create a bundled zip containing all models matching the base name
            File bundledZip = networksaver.createZipFromModels(modelDir, baseName, zipName);

            if (!bundledZip.exists() || bundledZip.length() == 0) {
                return ResponseEntity.notFound().build();
            }

            Path filePath = bundledZip.toPath().normalize();
            Resource resource = new UrlResource(filePath.toUri());

            return ResponseEntity.ok()
                    .contentType(MediaType.APPLICATION_OCTET_STREAM)
                    .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + bundledZip.getName() + "\"")
                    .body(resource);

        } catch (Exception e) {
            log.error("Failed to bundle and download models for base name: {}", baseName, e);
            return ResponseEntity.internalServerError().build();
        }
    }

}
