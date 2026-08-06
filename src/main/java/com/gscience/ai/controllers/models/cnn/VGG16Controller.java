package com.gscience.ai.controllers.models.cnn;


import com.gscience.ai.services.model.cnn.VGG16Service;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.Map;

@Log4j2
@RestController
@RequestMapping("/api/{version}/models/VGG6")
@RequiredArgsConstructor
@Tag(name = """
        Sample how to train and pretrained CNN model,  
        and refine  
        """)
public class VGG16Controller {

    private final VGG16Service vGG16Sv;

    @Operation(summary = "Upload two distinct CSV sheets to process corporate and metadata changes atomicly.")
    @PostMapping(value = "/train", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<Map<String, Object>> uploadDualDataStreams(
    ) throws IOException {

        vGG16Sv.train();


        return ResponseEntity.ok().build();
    }

    @PostMapping(value = "/detect-cat", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<String> detectedCat(
            @RequestParam("file")
            MultipartFile file,
            @RequestParam("threshold")
            Double threshold
    ){
        var result = vGG16Sv.detectCat(file,threshold);

        return ResponseEntity.ok(result.toString());
    }

}
