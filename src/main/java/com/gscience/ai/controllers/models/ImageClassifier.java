package com.gscience.ai.controllers.models;

import com.gscience.ai.services.image.classifier.ImageClassifierService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.Map;

@Log4j2
@RequiredArgsConstructor
@RestController
@RequestMapping("/api/{version}/image-classefier")
@Tag(name= "To classify images")
public class ImageClassifier {

    private final ImageClassifierService imageClassifierSv;

    @Operation(summary = "Upload two distinct CSV sheets to process corporate and metadata changes atomicly.")
    @PostMapping(value = "/train", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<Map<String, Object>> uploadDualDataStreams(
    ) throws IOException {

        imageClassifierSv.train();


        return ResponseEntity.ok(Map.of(
                "status", "SUCCESS",
                "message", "Both CSV transactional datasets synchronized without violations."
        ));
    }

}
