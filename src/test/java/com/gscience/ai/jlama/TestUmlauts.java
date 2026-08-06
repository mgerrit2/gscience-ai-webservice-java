package com.gscience.ai.jlama;

// De ontbrekende imports:

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.jlama.JlamaChatModel;
import org.junit.jupiter.api.Test;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.*;

public class TestUmlauts {


    @Test
    void firstTest() {
        ChatModel model = JlamaChatModel.builder()
                .modelName("tjake/Llama-3-8B-Instruct-Jlama-Q4")
                .temperature(0.1f) // Lagere temperatuur voor meer precisie
                .build();

        // Duidelijkere instructie om verwarring te voorkomen
        String prompt = "Translate the Dutch word 'Voetbal' to German. " +
                "Output ONLY the German word and then spell that German word.";

        String output = model.chat(prompt);

        // Gebruik UTF-8 voor de console output
        PrintStream out = new PrintStream(System.out, true, StandardCharsets.UTF_8);
        out.println("Model Output: " + output);

        assertNotNull(output);
        String lowerOutput = output.toLowerCase();

        // Check op inhoud in plaats van exacte match
        // We zoeken naar 'fußball' of 'fussball' (sommige modellen gebruiken 'ss')
        boolean correctTranslation = lowerOutput.contains("fußball") || lowerOutput.contains("fussball");

        // Check of hij de letters van het DUITSE woord spelde (F-U-S-S...)
        boolean correctSpelling = lowerOutput.contains("f-u-s");

        assertTrue(correctTranslation && correctSpelling,
                "De vertaling of spelling was niet zoals verwacht. Output: " + output);
    }



}
