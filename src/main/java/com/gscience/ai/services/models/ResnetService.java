package com.gscience.ai.services.models;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import net.coobird.thumbnailator.Thumbnails;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.util.*;

@RequiredArgsConstructor
@Log4j2
@Service
public class ResnetService {

    @Value("${onnx.model.path:models/resnet101.onnx}")
    private String modelPath;

    @Value("${onnx.labels.path:models/imagenet_classes.txt}")
    private String labelsPath;

    private OrtEnvironment env;
    private OrtSession session;

    // ImageNet normalization factors
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD = {0.229f, 0.224f, 0.225f};

    private List<String> labels = new ArrayList<>();

    // DTO Record for Prediction Output
    public record PredictionResult(int classId, String label, String probability) {}

    @PostConstruct
    public void init() {
        try {

            log.info("Initializing ONNX Runtime Environment...");

            this.env = OrtEnvironment.getEnvironment();

            // Load model from classpath (e.g., src/main/resources/models/resnet101.onnx)
            try (InputStream modelStream = getClass().getClassLoader().getResourceAsStream("models/resnet101.onnx")) {
                if (modelStream == null) {
                    throw new IllegalArgumentException("ONNX model file not found at: " + modelPath);
                }
                byte[] modelBytes = modelStream.readAllBytes();

                OrtSession.SessionOptions options = new OrtSession.SessionOptions();
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

                this.session = env.createSession(modelBytes, options);

                log.info("ONNX ResNet-101 session initialized successfully from classpath!");

            }

            // 2. Load ImageNet Labels
            try (InputStream labelStream = getClass().getClassLoader().getResourceAsStream(labelsPath)) {
                if (labelStream != null) {
                    try (BufferedReader reader = new BufferedReader(new InputStreamReader(labelStream, StandardCharsets.UTF_8))) {
                        this.labels = reader.lines().toList();
                    }
                    log.info("Loaded {} ImageNet labels.", labels.size());
                } else {
                    log.warn("Labels file not found at {}. Fallback to numeric class IDs only.", labelsPath);
                }
            }
        } catch (Exception e) {
            log.error("Failed to initialize ONNX ResNet model", e);
            throw new RuntimeException("Could not load ONNX model", e);
        }
    }

    /**
     * Accepts an InputStream of an image, preprocesses it, runs ONNX inference,
     * and returns the raw 1000-class logits.
     */
    public List<PredictionResult> classifyImage(InputStream imageInputStream, int topK) {
        try {
            BufferedImage originalImage = ImageIO.read(imageInputStream);
            if (originalImage == null) {
                throw new IllegalArgumentException("Invalid or unsupported image file");
            }

            BufferedImage croppedImage = preprocessImage(originalImage);
            FloatBuffer inputBuffer = imageToFloatBuffer(croppedImage);
            float[] logits = runInference(inputBuffer);

            return getTopKPredictions(logits, topK);

        } catch (Exception e) {
            log.error("Error during image processing/inference", e);
            throw new RuntimeException("Inference processing error", e);
        }
    }

    /**
     * Resizes the shortest edge to 256px and crops a 224x224 center patch.
     */
    private BufferedImage preprocessImage(BufferedImage image) throws Exception {
        int width = image.getWidth();
        int height = image.getHeight();

        int targetWidth, targetHeight;
        if (width < height) {
            targetWidth = 256;
            targetHeight = (int) ((double) height / width * 256);
        } else {
            targetHeight = 256;
            targetWidth = (int) ((double) width / height * 256);
        }

        // Step 1: Scale shortest side to 256px
        BufferedImage resizedImage = Thumbnails.of(image)
                .size(targetWidth, targetHeight)
                .asBufferedImage();

        // Step 2: Center crop to 224x224
        return Thumbnails.of(resizedImage)
                .size(224, 224)
                .crop(net.coobird.thumbnailator.geometry.Positions.CENTER)
                .asBufferedImage();
    }

    /**
     * Converts a 224x224 BufferedImage into an NCHW (1, 3, 224, 224) normalized FloatBuffer.
     */
    private FloatBuffer imageToFloatBuffer(BufferedImage image) {
        int width = 224;
        int height = 224;
        FloatBuffer buffer = FloatBuffer.allocate(1 * 3 * height * width);

        int[] rgbData = new int[width * height];
        image.getRGB(0, 0, width, height, rgbData, 0, width);

        // Populate Planar CHW Format (Red plane, Green plane, Blue plane)
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int pixel = rgbData[h * width + w];

                    // Extract color component (0..255)
                    int val = switch (c) {
                        case 0 -> (pixel >> 16) & 0xFF; // R
                        case 1 -> (pixel >> 8) & 0xFF;  // G
                        case 2 -> pixel & 0xFF;         // B
                        default -> 0;
                    };

                    // Scale to [0.0, 1.0] and Normalize: (val - mean) / std
                    float normalizedVal = ((val / 255.0f) - MEAN[c]) / STD[c];
                    buffer.put(normalizedVal);
                }
            }
        }

        buffer.rewind();
        return buffer;
    }

    /**
     * Executes a forward pass through the ONNX ResNet model using preprocessed image data.
     * <p>
     * This method wraps the normalized NCHW image buffer into an ONNX tensor with dimensions
     * {@code [1, 3, 224, 224]}, passes it into the active {@link OrtSession}, and extracts the
     * raw, unnormalized output logits.
     * </p>
     * <p>
     * Both the created {@link OnnxTensor} and the execution {@link OrtSession.Result} are managed
     * via try-with-resources blocks to guarantee native C++ memory handle disposal immediately
     * after the inference run.
     * </p>

     * @param inputBuffer a {@link FloatBuffer} containing a normalized NCHW (1x3x224x224) image tensor
     * @return a 1D float array of raw prediction logits corresponding to the ImageNet classes (size 1000)
     * @throws OrtException if an error occurs during tensor creation or within the ONNX execution engine
     */
    private float[] runInference(FloatBuffer inputBuffer) throws OrtException {
        long[] shape = {1, 3, 224, 224};

        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputBuffer, shape)) {
            Map<String, OnnxTensor> inputs = Map.of("input", inputTensor);

            try (OrtSession.Result results = session.run(inputs)) {
                float[][] outputs = (float[][]) results.get(0).getValue();
                return outputs[0]; // Logits array of size 1000
            }
        }
    }

    /**
     * Processes raw model logits to extract the top K class predictions with human-readable labels.
     * <p>
     * This method first normalizes the input logits into a probability distribution via
     * {@link #softmax(float[])}. It then organizes the probabilities using a max-heap
     * ({@link PriorityQueue}) to efficiently retrieve the {@code k} highest confidence predictions.
     * Each result is mapped to its human-readable ImageNet label and formatted as a percentage string.
     * </p>
     *
     * @param logits raw output tensor values from the ONNX model inference pass
     * @param k      the number of top prediction results to return
     * @return a list of {@link PredictionResult} records sorted in descending order of probability,
     *         containing class IDs, human-readable labels, and formatted percentage strings
     */
    private List<PredictionResult> getTopKPredictions(float[] logits, int k) {
        float[] probabilities = softmax(logits);

        PriorityQueue<Map.Entry<Integer, Float>> maxHeap = new PriorityQueue<>(
                (a, b) -> Float.compare(b.getValue(), a.getValue())
        );

        for (int i = 0; i < probabilities.length; i++) {
            maxHeap.offer(new AbstractMap.SimpleEntry<>(i, probabilities[i]));
        }

        List<PredictionResult> topK = new ArrayList<>();
        for (int i = 0; i < k && !maxHeap.isEmpty(); i++) {
            Map.Entry<Integer, Float> entry = maxHeap.poll();
            int classId = entry.getKey();
            float prob = entry.getValue();

            // Look up human-readable label
            String label = (classId < labels.size()) ? labels.get(classId) : "Unknown Class";
            String formattedProb = String.format(Locale.US, "%.2f%%", prob * 100);

            topK.add(new PredictionResult(classId, label, formattedProb));
        }

        return topK;
    }

    /**
     * Computes the numerically stable Softmax probability distribution over raw model logits.
     * <p>
     * This method applies the exponential function to each element after subtracting the maximum
     * value across all logits. Subtracting the maximum prevents numerical overflow
     * (e.g., returning {@link Float#NaN} or positive infinity when exponentiating large values).
     * The resulting exponentiated values are then normalized so that their sum equals 1.0.
     * </p>
     *
     * @param logits an array of raw, unnormalized prediction scores output by the neural network
     * @return a new array of normalized class probabilities bounded in the range [0.0, 1.0],
     *         where the sum of all elements equals 1.0
     */
    private float[] softmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float val : logits) {
            if (val > max) max = val;
        }

        float sum = 0.0f;
        float[] exp = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            exp[i] = (float) Math.exp(logits[i] - max); // Numerically stable softmax
            sum += exp[i];
        }

        for (int i = 0; i < logits.length; i++) {
            exp[i] /= sum;
        }
        return exp;
    }

    /**
     * Releases native memory and cleans up ONNX Runtime resources before bean destruction.
     * <p>
     * Annotated with {@link PreDestroy}, this lifecycle hook ensures that both the active
     * {@link OrtSession} and underlying {@link OrtEnvironment} native handles are closed
     * explicitly when the Spring application context stops, preventing native heap memory leaks.
     * </p>
     *
     * @see OrtSession#close()
     * @see OrtEnvironment#close()
     */
    @PreDestroy
    public void cleanup() {
        log.info("Cleaning up ONNX Runtime resources...");
        try {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
        } catch (OrtException e) {
            log.warn("Error closing ONNX Runtime session", e);
        }
    }

}
