/**
 * 
 */
package dlJavaMavenPackage1;

import java.io.File;
import java.io.IOException;
import java.util.Random;

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
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * @author pranaysadarangani
 *
 */
public class DLJavaMaven1 {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		System.out.println("test");
		
		int height = 28; 		// The number of rows of a matrix. - Image is 28x28 pixels
		int width = 28; 		// The number of columns of a matrix. - Image is 28x28 pixels
		
		int channels = 1;		// 	Grayscale implies single channel
		int rngseed = 123;		// 	This random-number generator applies a seed to ensure that 
		  						//	the same initial weights are used when training. 
		  						//	 We’ll explain why this matters later.
		
		Random randNumGen = new Random(rngseed);
		
		int numEpochs = 1; 	// An epoch is a complete pass through a given dataset.
		
		File trainData = new File(ClassLoader.getSystemResource("training").getPath());
		System.out.println(trainData.getPath());
		
		File testData = new File(ClassLoader.getSystemResource("testing").getPath());
		System.out.println(testData.getPath());
		
		FileSplit train = new FileSplit(trainData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);
		FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);
		
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
		
		ImageRecordReader recordReader = new ImageRecordReader(height,width,channels);
		
		try {
			recordReader.initialize(train);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		int batchSize = 128;	// How many examples to fetch with each step.
		int outputNum = 10; 	// Number of possible outcomes (e.g. labels 0 through 9).
		
		DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
		
        // Scale pixel values to 0-1
		
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	            .seed(rngseed)
	            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	            .iterations(1)
	            .learningRate(0.006)
	            //.updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS).momentum(0.9)
	            .updater(org.deeplearning4j.nn.conf.Updater.NESTEROVS)
	            .regularization(true).l2(1e-4)
	            .list()
	            .layer(0, new DenseLayer.Builder()
	                    .nIn(height * width)
	                    .nOut(100)
	                    .activation(Activation.RELU)
	                    .weightInit(WeightInit.XAVIER)
	                    .build())
	            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	                    .nIn(100)
	                    .nOut(outputNum)
	                    .activation(Activation.SOFTMAX)
	                    .weightInit(WeightInit.XAVIER)
	                    .build())
	            .pretrain(false).backprop(true)
	            .setInputType(InputType.convolutional(height,width,channels))
	            .build();

		
        
		
	}

}
