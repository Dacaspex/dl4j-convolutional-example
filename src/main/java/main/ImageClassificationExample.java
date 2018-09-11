package main;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

public class ImageClassificationExample {

    private static Logger logger = LoggerFactory.getLogger(ImageClassificationExample.class);

    public static void main(String[] args) throws Exception {
        logger.info("test");

        int width = 28;
        int height = 28;
        int channels = 1;
        int seed = 123;
        Random random = new Random(seed);
        int batchSize = 128;
        int outputNum = 5; // Number of labels
        int epochs = 15;

        // Get the data
        File trainingData = new File("data/training");
        File validationData = new File("data/validation");

        // File split
        FileSplit training = new FileSplit(trainingData, NativeImageLoader.ALLOWED_FORMATS, random);
        FileSplit validation = new FileSplit(validationData, NativeImageLoader.ALLOWED_FORMATS, random);

        // Extract label from parent path
        ParentPathLabelGenerator labelGenerator = new ParentPathLabelGenerator();

        // Record image reader
        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelGenerator);
        recordReader.initialize(training);
//        recordReader.setListeners(new LogRecordListener()); // Testing purposes

        // Dataset iterator
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);

        // Scale pixel values to [0, 1]
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(iterator);
        iterator.setPreProcessor(scaler);

        // Build our model
        logger.info("*** Build model ***");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(0.006, 0.9))
                .l2(0.0001)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(height * width)
                        .nOut(100)
                        .activation(new ActivationReLU())
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(100)
                        .nOut(outputNum)
                        .activation(new ActivationSoftmax())
                        .weightInit(WeightInit.XAVIER)
                        .build()
                )
                .pretrain(false)
                .backprop(true)
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.setIterationCount(1);
        model.init();

        model.setListeners(new ScoreIterationListener(10));

        logger.info("*** Training model ***");

        for (int i = 0; i < epochs; i++) {
            model.fit(iterator);
        }

        logger.info("*** Evaluate model ***");

        recordReader.reset();
        recordReader.initialize(validation);

        DataSetIterator testIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, outputNum);
        scaler.fit(testIterator);
        testIterator.setPreProcessor(scaler);

        // Create evaluation object
        Evaluation evaluation = new Evaluation(outputNum);

        while (testIterator.hasNext()) {
            DataSet next = testIterator.next();
            INDArray output = model.output(next.getFeatures());
            evaluation.eval(next.getLabels(), output);
        }

        logger.info(evaluation.stats());
    }
}
