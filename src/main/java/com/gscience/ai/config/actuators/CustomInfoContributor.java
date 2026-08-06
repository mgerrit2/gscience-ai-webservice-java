package com.gscience.ai.config.actuators;

import org.springframework.boot.actuate.info.Info;
import org.springframework.boot.actuate.info.InfoContributor;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;

@Component
public class CustomInfoContributor implements InfoContributor {

    @Override
    public void contribute(Info.Builder builder) {
        Map<String, Object> aiDetails = new HashMap<>();

        aiDetails.put("cuda", getCudoInfo());
        aiDetails.put("java", getJavaVersion());

        builder.withDetail("aiEnvironment", aiDetails);
    }

    /**
     * Retrieves the core runtime environment characteristics of the executing Java Virtual Machine.
     * <p>
     * This method extracts structured version tokens natively from the runtime engine via
     * {@link Runtime#version()} and safely serializes the output into a string. Converting
     * the object to a string is a critical step to prevent Jackson serialization failures
     * when rendering downstream Spring Actuator JSON payloads.
     * </p>
     *
     * @return a {@link Map} containing key-value diagnostic metadata describing the
     * current JVM deployment environment.
     * <ul>
     * <li>{@code "Full Version:"} - The complete, unparsed release string
     * (e.g., "21.0.11+10-LTS").</li>
     * </ul>
     * @see Runtime#version()
     */
    private Map<String, Object> getJavaVersion(){
        Map<String, Object> javaDetails = new HashMap<>();

        Runtime.Version version = Runtime.version();
        javaDetails.put("fullVersion", version.toString());
        javaDetails.put("major", version.feature());
        javaDetails.put("minor", version.interim());
        javaDetails.put("update", version.update());
        javaDetails.put("patch", version.patch());

        return javaDetails;
    }

    /**
     * Probes the system classpath and hardware state to gather CUDA and ND4J runtime environment metrics.
     * <p>
     * This method attempts to initialize and communicate with the underlying native matrix calculation
     * engine (e.g., via the {@code nd4j-cuda} hardware bridge). It captures the active backend implementation
     * and handles native linkage or driver deployment failures gracefully, ensuring that any initialization issues
     * do not cause cascading crashes during system diagnostics.
     * </p>
     * <p>
     * <i>Implementation Note:</i> If native binaries are missing or incompatible on the host OS,
     * ND4J will throw an instance of {@link java.lang.Error} (such as {@link java.lang.NoClassDefFoundError}).
     * Ensure your catch block intercepts {@link java.lang.Throwable} to isolate these native anomalies safely.
     * </p>
     *
     * @return a {@link Map} containing hardware execution platform details:
     * <ul>
     * <li>{@code "cuda.available"} - A {@link Boolean} flag indicating if the hardware backend initialized successfully.</li>
     * <li>{@code "nd4j.backend"} - A {@link String} denoting the active execution layer (e.g., {@code "JCublasBackend"}), present only if available.</li>
     * <li>{@code "error"} - A {@link String} outlining the root initialization exception trace if the backend registration failed.</li>
     * </ul>
     * @see org.nd4j.linalg.factory.Nd4j#getBackend()
     */
    private Map<String, Object> getCudoInfo(){
        Map<String, Object> cudaDetails = new HashMap<>();

        try {
            // Your existing ND4J code here (e.g., Nd4j.getBackend().toString())
            cudaDetails.put("nd4j.backend", org.nd4j.linalg.factory.Nd4j.getBackend().toString());
            cudaDetails.put("cuda.available", true);


        } catch (Exception t) {
            // Capture the initialization failure gracefully without crashing Actuator
            cudaDetails.put("cuda.available", false);
            cudaDetails.put("error", "ND4J Backend initialization failed: " + t.getMessage());
        }

        return cudaDetails;
    }

}
