package com.gmo.isto.dlwork;

import com.gmo.isto.dlwork.tools.DocItem;
import com.gmo.isto.dlwork.tools.LoadDataFromDB;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.UiServer;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Random;

/**GravesLSTM Character modelling example
 * @author Alex Black

   Example: Train a LSTM RNN to generates text, one character at a time.
	This example is somewhat inspired by Andrej Karpathy's blog post,
	"The Unreasonable Effectiveness of Recurrent Neural Networks"
	http://karpathy.github.io/2015/05/21/rnn-effectiveness/

	One minor difference between this example and Karpathy's work:
	The LSTM architectures appear to differ somewhat. GravesLSTM has peephole connections that
	Karpathy's char-rnn implementation appears to lack. See GravesLSTM javadoc for details.
	There are pros and cons to both architectures (addition of peephole connections is a more powerful
	model but has more parameters per unit), though they are not radically different in practice.

	This example is set up to train on the Complete Works of William Shakespeare, downloaded
	from Project Gutenberg. Training on other text sources should be relatively easy to implement.

    For more details on RNNs in DL4J, see the following:
    http://deeplearning4j.org/usingrnns
    http://deeplearning4j.org/lstm
    http://deeplearning4j.org/recurrentnetwork
 */
public class GravesLSTMCharModellingExample {
	public static void main( String[] args ) throws Exception {
		int lstmLayerSize = 200;					//Number of units in each GravesLSTM layer
		int miniBatchSize = 8;						//Size of mini batch to use when  training
		int exampleLength = 200;					//Length of each training example sequence to use. This could certainly be increased
        int tbpttLength = 40;                       //Length for truncated backpropagation through time. i.e., do parameter updates ever 50 characters
		int numEpochs = 3;							//Total number of training epochs
        int generateSamplesEveryNMinibatches = 4;  //How frequently to generate samples from the network? 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
		int nSamplesToGenerate = 4;					//Number of samples to generate after each training epoch
		int nWordsToSample = 200;				//Length of each sample to generate
		//String generationInitialization = null;		//Optional character initialization; a random character is used if null
		// Above is Used to 'prime' the LSTM with a character sequence to continue/complete.
		// Initialization characters must all be in com.gmo.isto.dlwork.CharacterIterator.getMinimalCharacterSet() by default
		Random rng = new Random(12345);

		UiServer server = UiServer.getInstance();
		System.out.println("Started on port " + server.getPort());

		//Get a DataSetIterator that handles vectorization of text into something we can use to train
		// our GravesLSTM network.
		//CharacterIterator iter = getShakespeareIterator(miniBatchSize,exampleLength);
		JapaneseWordIterator iter = getNewsIterator(miniBatchSize,exampleLength);
		int nOut = iter.totalOutcomes();

		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(3)
			.learningRate(0.01)
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.RMSPROP)
			.list()
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.activation("tanh").build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.activation("tanh").build())
			.layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
					.nIn(lstmLayerSize).nOut(nOut).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
			.pretrain(false).backprop(true)
			.build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		net.setListeners(new HistogramIterationListener(1));


		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);

		//Do training, and then generate and print samples from network
        int miniBatchNumber = 0;
		for( int i=0; i<numEpochs; i++ ){
			System.out.println("Starting Epoch: " + (i+1));
            while(iter.hasNext()){
                DataSet ds = iter.next();
                net.fit(ds);
                if(++miniBatchNumber % generateSamplesEveryNMinibatches == 0){
                    System.out.println("--------------------");
                    System.out.println("Completed " + miniBatchNumber + " minibatches of size " + miniBatchSize + "x" + exampleLength + " characters" );
                    //System.out.println("Sampling characters from network given initialization \"" + (generationInitialization == null ? "" : generationInitialization) + "\"");
                    String[] samples = sampleWordsFromNetwork(net,iter,rng,nWordsToSample,nSamplesToGenerate);
                    for( int j=0; j<samples.length; j++ ){
                        System.out.println("----- Sample " + j + " -----");
                        System.out.println(samples[j]);
                        System.out.println();
                    }
                }
            }

			iter.reset();	//Reset iterator for another epoch
		}

		System.out.println("\n\nExample complete");
	}

	/** Downloads Shakespeare training data and stores it locally (temp directory). Then set up and return a simple
	 * DataSetIterator that does vectorization based on the text.
	 * @param miniBatchSize Number of text segments in each training mini-batch
	 * @param sequenceLength Number of characters in each text segment.
	 */
	private static CharacterIterator getShakespeareIterator(int miniBatchSize, int sequenceLength) throws Exception{
		//The Complete Works of William Shakespeare
		//5.3MB file in UTF-8 Encoding, ~5.4 million characters
		//https://www.gutenberg.org/ebooks/100
		String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
		String tempDir = System.getProperty("java.io.tmpdir");
		String fileLocation = tempDir + "/Shakespeare.txt";	//Storage location from downloaded file
		File f = new File(fileLocation);
		if( !f.exists() ){
			FileUtils.copyURLToFile(new URL(url), f);
			System.out.println("File downloaded to " + f.getAbsolutePath());
		} else {
			System.out.println("Using existing text file at " + f.getAbsolutePath());
		}

		if(!f.exists()) throw new IOException("File does not exist: " + fileLocation);	//Download problem?

		char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
		return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
				miniBatchSize, sequenceLength, validCharacters, new Random(12345));
	}

	private static JapaneseWordIterator getNewsIterator(int miniBatchSize, int sequenceLength) throws Exception{
		String inputSql = "select id, post_content from xb_corpus where post_length < 2000 limit 150";
		//String inputSql = "select id, post_title from xb_corpus where post_length < 10000";

		List<DocItem> rs = LoadDataFromDB.loadDataFromSqlite(null, inputSql);
		return new JapaneseWordIterator(rs, miniBatchSize, sequenceLength, new Random(12345));
	}


	/** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
	 * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
	 * Note that the initalization is used for all samples
	 * @param wordsToSample Number of characters to sample from network (excluding initialization)
	 * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
	 * @param iter com.gmo.isto.dlwork.CharacterIterator. Used for going from indexes back to characters
	 */
	private static String[] sampleWordsFromNetwork(MultiLayerNetwork net,
												   JapaneseWordIterator iter, Random rng, int wordsToSample, int numSamples ){
		//Set up initialization. If no initialization: use a random character
		int N = 1;
		String []initialization = new String[N];
		for(int i=0; i<N; i++){
			initialization[i] = String.valueOf(iter.getRandomWord());
		}

		//Create input for initialization
		INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length);
		for( int i=0; i<initialization.length; i++ ){
			int idx = iter.convertWordToIndex(initialization[i]);
			for( int j=0; j<numSamples; j++ ){
				initializationInput.putScalar(new int[]{j,idx,i}, 1.0f);
			}
		}

		StringBuilder[] sb = new StringBuilder[numSamples];
		for( int i=0; i<numSamples; i++ ) {
			String line = null;
			sb[i] = new StringBuilder();
			for(int j=0; j<initialization.length;j++)
				sb[i].append(initialization[j]);

			sb[i].append(":\n");
		}

		//Sample from network (and feed samples back into input) one character at a time (for all samples)
		//Sampling is done in parallel here
		net.rnnClearPreviousState();
		INDArray output = net.rnnTimeStep(initializationInput);
		output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output

		for( int i=0; i<wordsToSample; i++ ){
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
			//Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
			for( int s=0; s<numSamples; s++ ){
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
				int sampledWordIdx = sampleFromDistribution(outputProbDistribution,rng);

				nextInput.putScalar(new int[]{s,sampledWordIdx}, 1.0f);		//Prepare next time step input
				sb[s].append(iter.convertIndexToWord(sampledWordIdx));	//Add sampled character to StringBuilder (human readable output)
			}

			output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
		}

		String[] out = new String[numSamples];
		for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
		return out;
	}

	/** Given a probability distribution over discrete classes, sample from the distribution
	 * and return the generated class index.
	 * @param distribution Probability distribution over classes. Must sum to 1.0
	 */
	private static int sampleFromDistribution( double[] distribution, Random rng ){
		double d = rng.nextDouble();
		double sum = 0.0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			if( d <= sum ) return i;
		}
		//Should never happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
}
