package com.gscience.ai.services.gpu;

import com.gscience.ai.dto.gpu.GpuDiagnosticsDTO;
import lombok.extern.log4j.Log4j2;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Service;

@Log4j2
@Service
public class GpuDiagnostics {

    public GpuDiagnosticsDTO checkForMultiGPU(){

        CudaEnvironment.getInstance().getConfiguration().allowMultiGPU(true);

        // Force initialization of the backend context
        INDArray dummy = Nd4j.create(new float[]{1.0f, 2.0f});

        // Check how many devices CUDA recognizes
        int numDevices = org.bytedeco.cuda.global.cudart.cudaGetDeviceCount(new int[1]);
        log.info(">>> Total CUDA Devices Detected: {}" ,numDevices);

        // Get the list/collection of available devices
        java.util.List<Integer> availableDevices = CudaEnvironment.getInstance().getConfiguration().getAvailableDevices();
        log.info(">>> Available Devices List: {}" ,availableDevices);

        // Check if multiple devices are available
        boolean multiGpuAllowed = availableDevices != null && availableDevices.size() > 1;
        log.info(">>> Multi-GPU Available/Allowed: {}", multiGpuAllowed);

        // Print current active device index
        int[] currentDevice = new int[1];
        org.bytedeco.cuda.global.cudart.cudaGetDevice(currentDevice);
        log.info(">>> Currently Active GPU Device Index: {}", currentDevice[0]);

        return GpuDiagnosticsDTO.builder()
                .numDevices(numDevices)
                .availableDevices(availableDevices)
                .multiGpuAllowed(multiGpuAllowed)
                .currentDevice(currentDevice[0])
                .build();

    }


}
