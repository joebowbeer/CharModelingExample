package com.joebowbeer.dl4j.recurrent.character;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.api.storage.StatsStorage;
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
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

/**
 * GravesLSTM Character modeling example
 *
 * @author Alex Black
 *
 * Example: Train a LSTM RNN to generates text, one character at a time. This example is somewhat
 * inspired by Andrej Karpathy's blog post, "The Unreasonable Effectiveness of Recurrent Neural
 * Networks" http://karpathy.github.io/2015/05/21/rnn-effectiveness/
 *
 * This example is set up to train on the Complete Works of William Shakespeare, downloaded from
 * Project Gutenberg. Training on other text sources should be relatively easy to implement.
 *
 * For more details on RNNs in DL4J, see the following: http://deeplearning4j.org/usingrnns
 * http://deeplearning4j.org/lstm http://deeplearning4j.org/recurrentnetwork
 */
public class GravesLSTMCharModellingExample {

  public static void main(String[] args) throws Exception {
    // Number of units in each GravesLSTM layer
    int lstmLayerSize = 200;
    // Size of mini batch to use when  training
    int miniBatchSize = 32;
//    // Length of each training example sequence to use. This could certainly be increased
//    int exampleLength = 1000;
    // Length for truncated backpropagation through time. i.e., update params every 50 characters
    // 1000 characters / 50 tbptt length: 20 parameter updates per minibatch
    int tbpttLength = 100;
    // Total number of training epochs
    int numEpochs = 20;
    // How frequently to generate samples from the network?
    int generateSamplesEveryNMinibatches = 4;
    // Number of samples to generate after each training epoch
    int nSamplesToGenerate = 3;
    // Length of each sample to generate
    //int nCharactersToSample = 30;
    // Optional character initialization; a random character is used if null.
    // Used to 'prime' the LSTM with a character sequence to continue/complete.
    // These characters must all be in CharacterIterator.getMinimalCharacterSet() by default
    String generationInitialization = "^";
    // Use Random with seed for reproducibility
    Random rng = new Random(12345);

    List<IterationListener> listeners = new ArrayList<>();

    // Initialize the user interface backend
    UIServer uiServer = UIServer.getInstance();
    // Configure where the network information is to be stored.
    // Use new FileStatsStorage(File) for saving and loading later
    StatsStorage statsStorage = new InMemoryStatsStorage();
    // Attach the StatsStorage instance to the UI.
    // This allows the contents of the StatsStorage to be visualized
    uiServer.attach(statsStorage);
    //Then add the StatsListener to collect this information from the network, as it trains
    listeners.add(new StatsListener(statsStorage));

    listeners.add(new ScoreIterationListener());

    // Get a DataSetIterator that handles vectorization of text into something we can use to train
    // our GravesLSTM network.
    DanceIterator iter = getDanceIterator(miniBatchSize, rng);
    int nOut = iter.totalOutcomes();

    // Set up network configuration:
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
        .learningRate(0.1)
        .rmsDecay(0.95)
        .seed(12345)
        .regularization(true)
        .l2(0.001)
        .weightInit(WeightInit.XAVIER)
        .updater(Updater.RMSPROP)
        .list()
        .layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
            .activation(Activation.TANH).build())
        .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
            .activation(Activation.TANH).build())
        // MCXENT + softmax for classification
        .layer(2, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation(Activation.SOFTMAX)
            .nIn(lstmLayerSize).nOut(nOut).build())
        .backpropType(BackpropType.TruncatedBPTT)
        .tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
        .pretrain(false).backprop(true)
        .build();

    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(listeners);

    // Print the  number of parameters in the network (and for each layer)
    Layer[] layers = net.getLayers();
    int totalNumParams = 0;
    for (int i = 0; i < layers.length; i++) {
      int nParams = layers[i].numParams();
      System.out.println("Number of parameters in layer " + i + ": " + nParams);
      totalNumParams += nParams;
    }
    System.out.println("Total number of network parameters: " + totalNumParams);

    // Do training, and then generate and print samples from network
    int miniBatchNumber = 0;
    for (int i = 0; i < numEpochs; i++) {
      while (iter.hasNext()) {
        net.fit(iter.next());
        if (++miniBatchNumber % generateSamplesEveryNMinibatches == 0) {
          System.out.println("--------------------");
          System.out.println("Completed " + miniBatchNumber + " minibatches of size "
              + miniBatchSize);
          System.out.println("Sampling characters from network given initialization \""
              + (generationInitialization == null ? "" : generationInitialization) + "\"");
          String[] samples = sampleCharactersFromNetwork(generationInitialization, net, iter, rng,
              nSamplesToGenerate);
          for (int j = 0; j < samples.length; j++) {
            System.out.println("----- Sample " + j + " -----");
            System.out.println(samples[j]);
            System.out.println();
          }
        }
      }

      iter.reset(); //Reset iterator for another epoch
    }

    System.out.println("\n\nExample complete");
    //uiServer.detach(statsStorage); // TODO?
  }

  /**
   * Downloads Shakespeare training data and stores it locally (temp directory). Then set up and
   * return a simple DataSetIterator that does vectorization based on the text.
   *
   * @param miniBatchSize Number of text segments in each training mini-batch
   * @param sequenceLength Number of characters in each text segment.
   * @param rng
   * @return 
   * @throws java.io.IOException 
   */
  public static CharacterIterator getShakespeareIterator(int miniBatchSize, int sequenceLength,
      Random rng) throws IOException {
    //The Complete Works of William Shakespeare
    //5.3MB file in UTF-8 Encoding, ~5.4 million characters
    //https://www.gutenberg.org/ebooks/100
    String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
    String tempDir = System.getProperty("java.io.tmpdir");
    // Storage location from downloaded file
    File folder = new File(tempDir, "dl4j-distribution-shakespeare");
    Charset encoding = Charset.forName("UTF-8");
    if (!folder.exists()) {
      File tmpFile = Files.createTempFile("Shakespeare", ".txt").toFile();
      tmpFile.deleteOnExit();
      FileUtils.copyURLToFile(new URL(url), tmpFile);
      System.out.println("File downloaded to " + tmpFile);
      // Copy segments into folder
      String s = FileUtils.readFileToString(tmpFile, encoding);
      for (int n = 0, i = 0; i + sequenceLength <= s.length(); n++, i += sequenceLength) {
        File f = new File(folder, String.valueOf(n));
        FileUtils.write(f, s.substring(i, i + sequenceLength), encoding);
      }
    } else {
      System.out.println("Using existing text files in " + folder.getAbsolutePath());
    }

    if (!folder.exists()) {
      // Download problem?
      throw new IOException("Folder does not exist: " + folder);
    }
    // Which characters are allowed? Others will be removed
    char[] validCharacters = CharacterIterator.getMinimalCharacterSet();
    return new CharacterIterator(folder, encoding, miniBatchSize, validCharacters, rng);
  }

  /**
   * Downloads American Country Dance training data and stores it locally (temp directory).
   * Then set up and return a simple DataSetIterator that does vectorization based on the text.
   *
   * @param miniBatchSize Number of text segments in each training mini-batch
   * @param rng
   * @return 
   * @throws java.io.IOException 
   */
  public static DanceIterator getDanceIterator(int miniBatchSize, Random rng)
      throws IOException {
    String url = "https://www.ibiblio.org/contradance/index/by_title.html";
    String tempDir = System.getProperty("java.io.tmpdir");
    // Storage location for downloaded file
    File file = new File(tempDir, "dl4j-rnn-ibiblio.html");
    Charset encoding = Charset.forName("UTF-8");
    if (!file.exists()) {
      FileUtils.copyURLToFile(new URL(url), file);
      System.out.println("File downloaded to " + file);
    } else {
      System.out.println("Using existing text in " + file);
    }

    if (!file.exists()) {
      // Download problem?
      throw new IOException("File does not exist: " + file);
    }
    // Which characters are allowed? Others will be removed
    char[] validCharacters = CharacterIterator.getDefaultCharacterSet();
    return new DanceIterator(file, encoding, miniBatchSize, validCharacters, rng);
  }

  /**
   * Generate a sample from the network, given an (optional, possibly null) initialization.
   * Initialization can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
   * Note that the initialization is used for all samples
   *
   * @param initialization String, may be null. If null, select a random character as initialization
   * for all samples
   * @param charactersToSample Number of characters to sample from network (excluding
   * initialization)
   * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
   * @param iter CharacterIterator. Used for going from indexes back to characters
   */
  private static String[] sampleCharactersFromNetwork(String initialization, MultiLayerNetwork net,
      DanceIterator iter, Random rng, int numSamples) {
    // Set up initialization. If no initialization: use a random character
    if (initialization == null) {
      initialization = String.valueOf(iter.getRandomCharacter());
    }

    // Create input for initialization
    INDArray initializationInput =
        Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
    char[] init = initialization.toCharArray();
    for (int i = 0; i < init.length; i++) {
      int idx = iter.convertCharacterToIndex(init[i]);
      for (int j = 0; j < numSamples; j++) {
        initializationInput.putScalar(new int[]{j, idx, i}, 1.0f);
      }
    }

    StringBuilder[] sb = new StringBuilder[numSamples];
    for (int i = 0; i < numSamples; i++) {
      sb[i] = new StringBuilder(initialization);
    }

    // Sample from network and feed samples back as input one character at a time (for all samples)
    // Sampling is done in parallel here
    net.rnnClearPreviousState();
    INDArray output = net.rnnTimeStep(initializationInput);
    // Gets the last time step output
    output = output.tensorAlongDimension(output.size(2) - 1, 1, 0);

    while(true) {
      // Set up next input (single time step) by sampling from previous output
      INDArray nextInput = Nd4j.zeros(numSamples, iter.inputColumns());
      // Output is a probability distribution.
      // Sample from this for each example we want to generate, and add it to the new input
      boolean appended = false;
      for (int s = 0; s < numSamples; s++) {
        if (sb[s].toString().endsWith("\n")) {
          continue;
        }
        double[] outputProbDistribution = new double[iter.totalOutcomes()];
        for (int j = 0; j < outputProbDistribution.length; j++) {
          outputProbDistribution[j] = output.getDouble(s, j);
        }
        int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution, rng);

        // Prepare next time step input
        nextInput.putScalar(new int[]{s, sampledCharacterIdx}, 1.0f);
        // Add sampled character to StringBuilder (human readable output)
        sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));
        appended = true;
      }

      if (!appended) {
        break;
      }
      output = net.rnnTimeStep(nextInput); // Do one time step of forward pass
    }

    String[] out = new String[numSamples];
    for (int i = 0; i < numSamples; i++) {
      out[i] = sb[i].substring(1);
    }
    return out;
  }

  /**
   * Given a probability distribution over discrete classes, sample from the distribution and return
   * the generated class index.
   *
   * @param distribution Probability distribution over classes. Must sum to 1.0
   * @param rng
   * @return 
   */
  public static int sampleFromDistribution(double[] distribution, Random rng) {
    double d = 0.0;
    double sum = 0.0;
    for (int t = 0; t < 10; t++) {
      d = rng.nextDouble();
      sum = 0.0;
      for (int i = 0; i < distribution.length; i++) {
        sum += distribution[i];
        if (d <= sum) {
          return i;
        }
      }
      // If we haven't found the right index yet, maybe the sum is slightly
      // lower than 1 due to rounding error, so try again.
    }
    // Should be extremely unlikely to happen if distribution is a valid probability distribution
    throw new IllegalArgumentException("Distribution is invalid? d=" + d + ", sum=" + sum);
  }
}
