package com.gscience.ai.components.image.classefier;

import com.gscience.ai.utility.ImageUtils;
import lombok.RequiredArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.springframework.stereotype.Component;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.AbstractMap;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

@Log4j2
@RequiredArgsConstructor
@Component
public class ImageFeatureExtractor {

    private final ImageUtils imageUtils;

    /**  Define RegEx to extract jpg name from the image class which is used to match against training labels */
    public  Pattern patt_get_jpg_name = Pattern.compile("[0-9]");

    /** Collects all images associated with a BizId. */
    public  List<Integer> getImgIdsFromBusinessId(Map<Integer, String> bizMap, List<String> businessIds) {
        return bizMap.entrySet().stream().filter(x -> businessIds.contains(x.getValue())).map(Map.Entry::getKey)
                .toList();
    }

    /** Get a list of images to load and process
     *
     * @param photoDir directory where the raw images reside
     * @param ids optional parameter to subset the images loaded from photoDir.
     */
    public  List<String> getImageIds(String photoDir, Map<Integer, String> businessMap, List<String> businessIds) {

        File d = new File(photoDir);

        // 1. Safety Check: Handle missing or invalid directory gracefully
        File[] files = d.listFiles();
        if (files == null) {
            throw new IllegalArgumentException("The photo directory does not exist or is invalid: " + photoDir);
        }

        // 2. Read all file paths safely
        List<String> imgsPath = Arrays.stream(files)
                .map(File::toString)
                .toList();

        // if businessMap of businessIds de specifieke waarde -1 bevatten, no filter toegepast
        boolean defaultBusinessMap = businessMap.size() == 1 && businessMap.get(-1).equals("-1");
        boolean defaultBusinessIds = businessIds.size() == 1 && businessIds.get(0).equals("-1");

        if (defaultBusinessMap || defaultBusinessIds) {

            return imgsPath;

        } else {

            // create map
            Map<Integer, String> imgsMap = imgsPath.stream()
                    .map(x -> new AbstractMap.SimpleEntry<Integer, String>(extractInteger(x), x))
                    .collect(Collectors.toMap(AbstractMap.SimpleEntry::getKey, AbstractMap.SimpleEntry::getValue));

            List<Integer> imgsPathSub = getImgIdsFromBusinessId(businessMap, businessIds);

            return imgsPathSub.stream().filter(imgsMap::containsKey).map(imgsMap::get)
                    .toList();

        }
    }

    /** Read and process images into a photoID -> vector map
     *
     * @param imgs list of images to read-in.  created from getImageIds function.
     * @param resizeImgDim dimension to rescale square images to
     * @param nPixels number of pixels to maintain.  mainly used to sample image to drastically reduce runtime while testing features.
     *
     */
    public  Map<Integer, List<Integer>> processImages(List<String> imgs, int resizeImgDim, int nPixels) {
        Function<String, AbstractMap.Entry<Integer, List<Integer>>> handleImg = x -> {
            BufferedImage img = null;
            try {
                img = ImageIO.read(new File(x));
            } catch (IOException e) {
                log.error(e);
            }
            img = imageUtils.makeSquare(img);
            img = imageUtils.resizeImg(img, resizeImgDim, resizeImgDim);
            List<Integer> value = imageUtils.image2gray(img);
            if(nPixels != -1) {
                value = value.subList(0, nPixels);
            }
            return new AbstractMap.SimpleEntry<Integer, List<Integer>>(extractInteger(x), value);
        };

        return imgs.stream().map(handleImg).filter(e -> !e.getValue().isEmpty())
                .collect(Collectors.toMap(e -> e.getKey(), e -> e.getValue()));
    }

    public  Map<Integer, List<Integer>> processImages(List<String> imgs, int resizeImgDim) {
        return processImages(imgs, resizeImgDim, -1);
    }

    public Map<Integer, List<Integer>> processImages(List<String> imgs) {
        return processImages(imgs, 128);
    }

    /**
     * Extraheert een numerieke waarde uit een gegeven bestandspad of bestandsnaam.
     * <p>
     * Deze methode scant het opgegeven pad met behulp van de vooraf gedefinieerde
     * reguliere expressie {@code patt_get_jpg_name}. Alle gevonden overeenkomsten
     * worden achter elkaar geplakt en vervolgens geconverteerd naar een {@link Integer}.
     * </p>
     *
     * @param path Het volledige bestandspad of de bestandsnaam waaruit het getal geëxtraheerd moet worden.
     * @return De geëxtraheerde waarde als een {@I_ntege_r}.
     * @throws NumberFormatException Als de resulterende string geen geldige weergave is van een getal,
     * leeg is, of de maximale waarde van een integer overschrijdt.
     * @throws NullPointerException Als {@code path} {@code null} is.
     */
    private  Integer extractInteger(String path) {
        StringBuilder sb = new StringBuilder();
        Matcher m = patt_get_jpg_name.matcher(path);
        while (m.find()) {
            sb.append(m.group());
        }
        return Integer.parseInt(sb.toString());
    }
}
