package com.gscience.ai.model.vgg;

import lombok.extern.log4j.Log4j2;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

@Log4j2
class VGGSpec {

    @Test
    void vggToZipfile() throws IOException {

        ZooModel model = VGG16.builder().build();
        log.info("Start Downloading VGG16 model...");

        ComputationGraph preTrainedNet = (ComputationGraph) model.initPretrained(PretrainedType.IMAGENET);

        File imagenetZip = new File("/home/mgerrit2/dev-ubuntu/software-projects-ubuntu/data/model/predefined/vgg16_imagenet.zip");
        boolean saveUpdater = false; // Set to true if you plan to continue training
        ModelSerializer.writeModel(preTrainedNet, imagenetZip, saveUpdater);

    }

}
