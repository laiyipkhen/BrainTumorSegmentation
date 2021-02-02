package ai.certifai.solution.segmentation.braintumor;

import ai.certifai.Helper;
import ai.certifai.solution.segmentation.CustomLabelGenerator;
import ai.certifai.solution.segmentation.car.CarDataSetIterator;
import ai.certifai.utilities.DataUtilities;
import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class BraintumorDatasetIterator {
    //private static final Logger log = org.slf4j.LoggerFactory.getLogger(CarDataSetIterator.class);
    private static final int height = 224;
    private static final int width = 224;
    private static final int channels = 1;
    private static final long seed = 12345;
    private static final Random random = new Random(seed);
    private static String inputDir;
    private static String downloadLink;
    private static InputSplit trainData, valData;
    private static int batchSize;

    private static List<Pair<String, String>> replacement = Arrays.asList(
            new Pair<>("inputs", "Mask"),
            new Pair<>(".tif", ".tif")
    );
    private static CustomLabelGenerator labelMaker = new CustomLabelGenerator(height, width, channels, replacement);

    //scale input to 0 - 1
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static ImageTransform transform;

    public BraintumorDatasetIterator() {
    }

    //This method instantiates an ImageRecordReader and subsequently a RecordReaderDataSetIterator based on it
    private static RecordReaderDataSetIterator makeIterator(InputSplit split) throws IOException {


        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);


//        Both train and val iterator need the preprocessing of converting RGB to Grayscale
        recordReader.initialize(split, transform);
        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 1, true);
        iter.setPreProcessor(scaler);

        return iter;


    }
    public static RecordReaderDataSetIterator trainIterator() throws IOException {
        return makeIterator(trainData);
    }

    public static RecordReaderDataSetIterator valIterator() throws IOException {
        return makeIterator(valData);
    }

    public static void setup(int batchSizeArg, int trainPerc, ImageTransform imageTransform) throws IOException {
        transform = imageTransform;
        setup(batchSizeArg, trainPerc);
    }
    public static void setup(int batchSizeArg, int trainPerc) throws IOException {

        inputDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        //    File dataZip = new File(Paths.get(inputDir, "data-science-bowl-2018", "data-science-bowl-2018.zip").toString());
        File classFolder = new File(Paths.get(inputDir, "BrainTumor").toString());





        batchSize = batchSizeArg;

        inputDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.data")
        ).toString();

        File imagesPath = new File(Paths.get(inputDir, "BrainTumorTrain", "inputs").toString());
        FileSplit imageFileSplit = new FileSplit(imagesPath, NativeImageLoader.ALLOWED_FORMATS, random);
        BalancedPathFilter imageSplitPathFilter = new BalancedPathFilter(random, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] imagesSplits = imageFileSplit.sample(imageSplitPathFilter, trainPerc, 100 - trainPerc);

        trainData = imagesSplits[0];
        //valData = imagesSplits[1];

        File imagesPathVal = new File(Paths.get(inputDir, "BrainTumorTest", "inputs").toString());
        FileSplit imageFileSplitVal = new FileSplit(imagesPathVal, NativeImageLoader.ALLOWED_FORMATS, random);
        BalancedPathFilter imageSplitPathFilterVal = new BalancedPathFilter(random, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] imagesSplitsVal = imageFileSplitVal.sample(imageSplitPathFilterVal, trainPerc, 100 - trainPerc);

        //trainData = imagesSplits[0];
        valData = imagesSplitsVal[0];
    }
}
