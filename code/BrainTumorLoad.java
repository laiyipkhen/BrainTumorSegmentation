package ai.certifai.solution.segmentation.braintumor;

import ai.certifai.Helper;
import ai.certifai.utilities.Visualization;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.swing.*;
import java.io.File;
import java.nio.file.Paths;

public class BrainTumorLoad {
    private static String modelImportDir;


    private static Logger log = LoggerFactory.getLogger(BrainTumorLoad.class);

    public static void main(String[] args) throws Exception
    {
        modelImportDir = Paths.get(
                System.getProperty("user.home"),
                Helper.getPropValues("dl4j_home.generated-models")
        ).toString();

        File modelSave =  new File(Paths.get(modelImportDir, "/BrainTumor.zip").toString());

        if(modelSave.exists() == false)
        {
            System.out.println("Model not exist. Abort");
            return;
        }

        /*
		#### LAB STEP 1 #####
		Load the saved model
        */



        // Set listeners
        StatsStorage statsStorage = new InMemoryStatsStorage();
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        uiServer.attach(statsStorage);

        // STEP 3: Load data into RecordReaderDataSetIterator
        BraintumorDatasetIterator.setup(1, 1);

        //Create iterators
        RecordReaderDataSetIterator imageDataSetVal = BraintumorDatasetIterator.valIterator();
        ComputationGraph model = ModelSerializer.restoreComputationGraph(modelSave, false);

        // VALIDATION
        Evaluation eval = new Evaluation(2);

        // VISUALISATION -  validation
        JFrame frameVal = Visualization.initFrame("Viz");
        JPanel panelVal = Visualization.initPanel(
                frameVal,
                1,
                224,
                224,
                1
        );

        // EXPORT IMAGES
        File exportDir = new File("export");

        if (!exportDir.exists()) {
            exportDir.mkdir();
        }

        float IOUtotal = 0;
        int count = 0;
        int count1=1;

        while (imageDataSetVal.hasNext()) {

            System.out.println(count1++);

            DataSet imageSetVal = imageDataSetVal.next();
            INDArray predict = model.output(imageSetVal.getFeatures())[0];
            INDArray labels = imageSetVal.getLabels();

            if (count % 5 == 0) {
                Visualization.export(exportDir, imageSetVal.getFeatures(), imageSetVal.getLabels(), predict, count);
            }

            count++;

            eval.eval(labels, predict);
            log.info(eval.stats());

            //STEP 5: Complete the code for IOU calculation here
            //Intersection over Union:  TP / (TP + FN + FP)
            float IOUNuclei = (float) eval.truePositives().get(1) / ((float) eval.truePositives().get(1) + (float) eval.falsePositives().get(1) + (float) eval.falseNegatives().get(1));
            IOUtotal = IOUtotal + IOUNuclei;

            System.out.println("IOU Brain Tumor " + String.format("%.3f", IOUNuclei));

            eval.reset();

            for (int n = 0; n < imageSetVal.asList().size(); n++) {
                Visualization.visualize(
                        imageSetVal.get(n).getFeatures(),
                        imageSetVal.get(n).getLabels(),
                        predict,
                        frameVal,
                        panelVal,
                        1,
                        224,
                        224
                );
            }
        }

        System.out.print("Mean IOU: " + IOUtotal / count);



    }

}
