package com.gscience.ai.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;
import org.springframework.web.servlet.config.annotation.ApiVersionConfigurer;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Override
    public void configureApiVersioning(ApiVersionConfigurer configurer) {
        configurer
                .addSupportedVersions("1.0", "2.0")
                .setDefaultVersion("1.0")
                // Allows requests without an explicit version (like /v3/api-docs)
                // to fall back to the defaultVersion ("1.0") instead of returning 400
                .setVersionRequired(false)
                .useVersionResolver(request -> {
                    String uri = request.getRequestURI();

                    // Only extract version from segment 1 if path starts with /api/
                    if (uri.startsWith("/api/")) {
                        String[] segments = uri.split("/");
                        // URI: /api/1.0/endpoint -> segments: ["", "api", "1.0", "endpoint"]
                        return (segments.length > 2) ? segments[2] : null;
                    }

                    // Non-API paths (Swagger, index.html) return null and default to "1.0"
                    return null;
                });
    }

    @Bean
    public RestClient.Builder restClientBuilder() {
        return RestClient.builder();
    }

}