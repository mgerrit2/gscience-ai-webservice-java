package com.gscience.ai.components.image.classefier;

import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;


/**
 * Component responsible for managing deep learning model persistence operations,
 * including saving, loading, listing, and archiving Deeplearning4j models.
 * Note: Direct ONNX export/conversion is currently not supported in this version.
 * <p>
 * ONNX is still not support in the .onnx version
 */
@Log4j2
@RequiredArgsConstructor
@Component
public class Networksaver {

    /**
     * Saves a trained MultiLayerNetwork model to the specified file path using Deeplearning4j's ModelSerializer.
     * * @param model       The trained MultiLayerNetwork instance to persist.
     * @param filePath    The target destination path where the model file will be stored (e.g., "results/models/model0.zip").
     * @param saveUpdater Flag indicating whether to save the optimizer and updater state (set to true if resuming training later).
     * @throws IOException If directory creation or file writing fails.
     */
    public void saveModel(MultiLayerNetwork model, String filePath, boolean saveUpdater) throws IOException {
        File locationToSave = new File(filePath);
        // Ensure parent directories exist
        if (locationToSave.getParentFile() != null) {
            locationToSave.getParentFile().mkdirs();
        }
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        log.info("Model successfully saved to: {}", filePath);
    }

    /**
     * Loads a serialized MultiLayerNetwork model from disk.
     * * @param filePath The file path to the saved model archive.
     * @return The restored MultiLayerNetwork instance.
     * @throws IOException If the file cannot be read or is invalid.
     */
    public MultiLayerNetwork loadModel(String filePath) throws IOException {
        File locationToLoad = new File(filePath);
        log.info("Loading model from: {}", filePath);
        return ModelSerializer.restoreMultiLayerNetwork(locationToLoad);
    }

    /**
     * Scans the designated models directory and retrieves a list of identifiers for all saved models.
     * * @param directoryPath The path of the directory containing model files.
     * @return A list of model identifiers with their file extensions stripped.
     */
    public List<String> getAllSavedModelNames(String directoryPath) {
        List<String> modelNames = new ArrayList<>();
        File dir = new File(directoryPath);

        if (dir.exists() && dir.isDirectory()) {
            File[] files = dir.listFiles((d, name) -> name.endsWith(".zip"));
            if (files != null) {
                for (File file : files) {
                    // Strip the .zip extension to get the model identifier
                    String name = file.getName();
                    modelNames.add(name.substring(0, name.lastIndexOf('.')));
                }
            }
        }
        return modelNames;
    }

    /**
     * Finds and filters saved model names matching a specific base naming pattern.
     * * @param directoryPath   The path of the directory to scan.
     * @param baseNamePattern The prefix pattern to match (e.g., "model").
     * @return A list of model names matching the base pattern and optional numeric suffixes.
     */
    public List<String> getModelsByBaseName(String directoryPath, String baseNamePattern) {
        List<String> matchedModels = new ArrayList<>();
        File dir = new File(directoryPath);

        if (dir.exists() && dir.isDirectory()) {
            File[] files = dir.listFiles((d, name) -> name.endsWith(".zip"));
            if (files != null) {
                for (File file : files) {
                    String fileName = file.getName();
                    String nameWithoutExtension = fileName.substring(0, fileName.lastIndexOf('.'));

                    // Matches if the file starts with the base pattern (e.g., "model")
                    if (nameWithoutExtension.matches(baseNamePattern + "\\d*")) {
                        matchedModels.add(nameWithoutExtension);
                    }
                }
            }
        }
        return matchedModels;
    }

    /**
     * Bundles all model files matching a specific base name pattern into a single compressed ZIP archive.
     * * @param directoryPath       The directory path where models are stored.
     * @param baseNamePattern     The base name pattern to filter models for bundling.
     * @param zipOutputFileName   The file name of the resulting archive bundle.
     * @return A File object pointing to the newly generated ZIP archive.
     * @throws IOException If an error occurs during file reading or archive compression.
     */
    public File createZipFromModels(String directoryPath, String baseNamePattern, String zipOutputFileName) throws IOException {
        File dir = new File(directoryPath);
        File zipFile = new File(directoryPath, zipOutputFileName);

        try (FileOutputStream fos = new FileOutputStream(zipFile);
             ZipOutputStream zos = new ZipOutputStream(fos)) {

            if (dir.exists() && dir.isDirectory()) {
                File[] files = dir.listFiles((d, name) -> name.endsWith(".zip"));
                if (files != null) {
                    for (File file : files) {
                        String fileName = file.getName();
                        String nameWithoutExtension = fileName.substring(0, fileName.lastIndexOf('.'));

                        // Match base name pattern (e.g., "model" matches model0, model1...)
                        if (nameWithoutExtension.matches(baseNamePattern + "\\d*")) {
                            ZipEntry zipEntry = new ZipEntry(fileName);
                            zos.putNextEntry(zipEntry);
                            Files.copy(file.toPath(), zos);
                            zos.closeEntry();
                        }
                    }
                }
            }
        }
        return zipFile;
    }

}
