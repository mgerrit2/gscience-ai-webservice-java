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

}
