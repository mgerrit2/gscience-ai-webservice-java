package com.gscience.ai.deeplearning4j.projects.titanic.survival.util;

import org.apache.spark.sql.SparkSession;

public final class SparkSessionUtil {
    private static volatile SparkSession INSTANCE;

    public static SparkSession getInstance() {
        if (null == INSTANCE) {
            synchronized (SparkSessionUtil.class) {
                if (null == INSTANCE)
                    INSTANCE = SparkSession
                            .builder()
                            .master("local[*]") // use all local cores
                            .config("spark.ui.enabled", "false") // Bypasses the Javax/Jakarta Servlet error
                            .config("spark.driver.memory", "4g")
                            .config("spark.driver.memory", "10g") // Increase memory for GPU overhead
                            // Enable GPU Discovery
                            // GPU Discovery (Bypasses the crash with your .bat file)
                            .config("spark.driver.resource.gpu.amount", "1")
                            .config("spark.driver.resource.gpu.discoveryScript", "./getGpusResources.bat")

                            .appName("SurvivalPredictionMLP")
                            .getOrCreate();

            }
        }

        return INSTANCE;
    }
}
