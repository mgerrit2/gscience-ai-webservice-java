package com.gscience.ai.jlama;

import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.jlama.JlamaChatModel;
import org.junit.jupiter.api.Test;

public class TestHelloworld {

    @Test
    void hello_World() {
        ChatModel model = JlamaChatModel.builder()
                .modelName("tjake/TinyLlama-1.1B-Chat-v1.0-Jlama-Q4")
                .build();

        String response = model.chat("Say 'Hello World'");
        System.out.println(response);
    }

}
