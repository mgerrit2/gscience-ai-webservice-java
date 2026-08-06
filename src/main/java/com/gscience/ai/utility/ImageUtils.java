package com.gscience.ai.utility;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.function.Function;

import org.imgscalr.Scalr;
import org.nd4j.common.primitives.Triple;
import org.springframework.stereotype.Component;

@Component
public class ImageUtils {

    // image 2 vector processing
    private Integer pixels2gray(Integer red, Integer green, Integer blue){
        return (red + green + blue) / 3;
    }

    private List<Integer> pixels2color(Integer red, Integer green, Integer blue) {
        return Arrays.asList(red, green, blue);
    }

    private <T> List<T> image2vec(BufferedImage img, Function<Triple<Integer, Integer, Integer>, T> f) {
        int w = img.getWidth();
        int h = img.getHeight();
        ArrayList<T> result = new ArrayList<>();
        for (int w1 = 0; w1 < w; w1++ ) {
            for (int h1 = 0; h1 < h; h1++) {
                int col = img.getRGB(w1, h1);
                int red =  (col & 0xff0000) / 65536;
                int green = (col & 0xff00) / 256;
                int blue = (col & 0xff);
                result.add(f.apply(new Triple<>(red, green, blue)));
            }
        }
        return result;
    }

    public List<Integer> image2gray(BufferedImage img) {
        return image2vec(img, t -> pixels2gray(t.getFirst(), t.getSecond(), t.getThird()));
    }

    public List<Integer> image2color(BufferedImage img) {
        return image2vec(img, t -> pixels2color(t.getFirst(), t.getSecond(), t.getThird()))
                .stream()
                .flatMap(Collection::stream)
                .toList();
    }

    // make image square
    public BufferedImage makeSquare(BufferedImage img) {

        int w = img.getWidth();
        int h = img.getHeight();
        int dim = Math.min(w, h);

        if (w == h) {
            return img;
        } else if (w > h) {
            return Scalr.crop(img, (w-h)/2, 0, dim, dim);
        } else  {
            return Scalr.crop(img, 0, (h-w)/2, dim, dim);
        }

    }

    // resize pixels
    public BufferedImage resizeImg(BufferedImage img, int width, int height) {
        return Scalr.resize(img, Scalr.Method.BALANCED, width, height);
    }

}
