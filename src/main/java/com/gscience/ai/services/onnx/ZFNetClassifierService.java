package com.gscience.ai.services.onnx;

import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

/**
 * https://www.youtube.com/watch?v=hWosZTj327c
 */
@RequiredArgsConstructor
@Log4j2
@Service
public class ZFNetClassifierService {

    @Value("${data.path.onnx.zfnet512-12.onnx}")
    String zfnet512Path;


}
