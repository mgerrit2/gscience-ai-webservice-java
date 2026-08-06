package com.gscience.ai.config;

import jakarta.annotation.PostConstruct;
import org.springdoc.core.utils.SpringDocUtils;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OpenAPIConfig {

    @PostConstruct
    public void ignoreComplexTypes() {
        // Ignore when used as parameters or request payloads
        SpringDocUtils.getConfig().addRequestWrapperToIgnore(
                org.apache.spark.sql.Dataset.class,
                org.nd4j.linalg.api.ndarray.INDArray.class
        );

        // IGNORE when used as return types/responses (CRITICAL FOR SPARK/DL4J)
        SpringDocUtils.getConfig().addResponseTypeToIgnore(
                org.apache.spark.sql.Dataset.class
        );
        SpringDocUtils.getConfig().addResponseTypeToIgnore(
                org.nd4j.linalg.api.ndarray.INDArray.class
        );

        // OPTIONAL: Map heavy types to simple String/Object schemas in OpenAPI docs
        SpringDocUtils.getConfig().replaceWithClass(
                org.apache.spark.sql.Dataset.class, Object.class
        );
        SpringDocUtils.getConfig().replaceWithClass(
                org.nd4j.linalg.api.ndarray.INDArray.class, Object.class
        );
    }

}
