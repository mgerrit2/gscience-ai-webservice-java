package com.gscience.ai.NLP;


import com.gscience.ai.GScienceAiApplication;
import opennlp.tools.sentdetect.SentenceDetectorME;
import opennlp.tools.sentdetect.SentenceModel;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import org.junit.jupiter.api.Test;

public class NLPSimpleSpec {

    @Test
    void name() {

        String paragraph = "Apache OpenNLP supports the most common NLP tasks, "
                + "such as tokenization, sentence segmentation, and part-of-speech tagging. "
                + "It's a powerful tool. Learn more at opennlp.apache.org.";

        // Load the sentence detection model
        try (InputStream modelIn = GScienceAiApplication.class.getResourceAsStream("/nlp/opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin")) {
            if (modelIn == null) {
                System.err.println("Error: en-sent.bin model not found in resources. Please download it.");
                return;
            }
            SentenceModel model = new SentenceModel(modelIn);
            SentenceDetectorME sentenceDetector = new SentenceDetectorME(model);

            // Detect sentences
            String[] sentences = sentenceDetector.sentDetect(paragraph);

            System.out.println("Original Paragraph:\n" + paragraph);
            System.out.println("\nDetected Sentences:");
            for (int i = 0; i < sentences.length; i++) {
                System.out.println("Sentence " + (i + 1) + ": " + sentences[i]);
            }


        } catch (IOException e) {
            throw new RuntimeException(e);
        }


    }
}