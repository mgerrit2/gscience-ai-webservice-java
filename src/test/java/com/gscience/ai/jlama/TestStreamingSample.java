package com.gscience.ai.jlama;

import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.jlama.JlamaStreamingChatModel;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;

public class TestStreamingSample {

    /**
     * https://docs.langchain4j.dev/integrations/language-models/jlama
     */
    @Disabled
    @Test
    void name() {

        StreamingChatModel model = JlamaStreamingChatModel.builder()
                .modelName("tjake/TinyLlama-1.1B-Chat-v1.0-Jlama-Q4")
                .build();

        CompletableFuture<ChatResponse> futureResponse = new CompletableFuture<>();
        model.chat("Tell me a joke about Java", new StreamingChatResponseHandler() {

            @Override
            public void onPartialResponse(String partialResponse) {
                System.out.print(partialResponse);
            }

            @Override
            public void onCompleteResponse(ChatResponse completeResponse) {
                futureResponse.complete(completeResponse);
            }

            @Override
            public void onError(Throwable error) {
                futureResponse.completeExceptionally(error);
            }
        });

        futureResponse.join();

    }


}
