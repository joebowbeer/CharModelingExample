package com.joebowbeer.dl4j.recurrent.character;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 * A simple DataSetIterator for use in the GravesLSTMCharModellingExample. Given a folder of text
 * files and a few options, generate feature vectors and labels for training, where we want to
 * predict the next character in the sequence.<br>
 * This is done by randomly choosing a a text file to start each sequence. Then we convert each
 * character to an index, i.e., a one-hot vector. Then the character 'a' becomes [1,0,0,0,...],
 * 'b' becomes [0,1,0,0,...], etc
 *
 * Feature vectors and labels are both one-hot vectors of the same length
 *
 * @author Alex Black
 */
public class CharacterIterator implements DataSetIterator {
  private final List<File> files;
  private Charset textFileEncoding;
  // Size of each minibatch (number of examples)
  private final int miniBatchSize;
  // Valid characters
  private final char[] validCharacters;
  private final Random rng;
  // Maps each character to an index ind the input/output
  private final Map<Character, Integer> charToIdxMap = new HashMap<>();
  private final Pattern invalidCharsPattern;

  private int cursor;

  /**
   * @param folder Folder containing text files to use for generating samples
   * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
   * @param miniBatchSize Number of examples per mini-batch
   * @param validCharacters Character array of valid characters. Characters not present in this
   * array will be removed
   * @param rng Random number generator, for repeatability if required
   * @throws IOException If text file cannot be loaded
   */
  public CharacterIterator(File folder, Charset textFileEncoding, int miniBatchSize,
      char[] validCharacters, Random rng) throws IOException {
    if (!folder.exists()) {
      throw new IOException("Could not access file (does not exist): " + folder);
    }
    if (miniBatchSize <= 0) {
      throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
    }

    this.files = Arrays.asList(folder.listFiles());
    this.textFileEncoding = textFileEncoding;
    this.miniBatchSize = miniBatchSize;
    this.validCharacters = validCharacters;
    this.rng = rng;

    // Store valid characters is a map for later use in vectorization
    for (int n = validCharacters.length; --n >= 0; ) {
      charToIdxMap.put(validCharacters[n], n);
    }

    // Create regex that matches invalid characters
    String escaped = new String(validCharacters).replaceAll("[\\W]", "\\\\$0");
    String regex = new StringBuilder("[^").append(escaped).append("]").toString();
    invalidCharsPattern = Pattern.compile(regex);

    reset();
  }

  /**
   * A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc
   * @return 
   */
  public static char[] getMinimalCharacterSet() {
    StringBuilder validChars = new StringBuilder();
    for (char c = 'a'; c <= 'z'; c++) {
      validChars.append(c);
    }
    for (char c = 'A'; c <= 'Z'; c++) {
      validChars.append(c);
    }
    for (char c = '0'; c <= '9'; c++) {
      validChars.append(c);
    }
    validChars.append(new char[]{
      '!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'
    });
    return validChars.toString().toCharArray();
  }

  /**
   * As per getMinimalCharacterSet(), but with a few extra characters
   * @return 
   */
  public static char[] getDefaultCharacterSet() {
    StringBuilder validChars = new StringBuilder();
    validChars.append(getMinimalCharacterSet());
    validChars.append(new char[]{
        '@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_', '\\', '|', '<', '>'
    });
    return validChars.toString().toCharArray();
  }

  public char convertIndexToCharacter(int idx) {
    return validCharacters[idx];
  }

  public int convertCharacterToIndex(char c) {
    return charToIdxMap.get(c);
  }

  public char getRandomCharacter() {
    return validCharacters[rng.nextInt(validCharacters.length)];
  }

  @Override
  public boolean hasNext() {
    return numExamples() > cursor;
  }

  @Override
  public DataSet next() {
    return next(miniBatchSize);
  }

  @Override
  public DataSet next(int num) {
    if (!hasNext()) {
      throw new NoSuchElementException();
    }
    try {
      return nextDataSet(num);
    } catch(IOException ex){
      throw new RuntimeException(ex);
    }
  }

  private DataSet nextDataSet(int num) throws IOException {
    int maxLength = 0;
    List<String> batch = new ArrayList<>(num);
    for (int n = num; --n >= 0 && hasNext(); cursor++) {
      List<String> lines = Files.readAllLines(files.get(cursor).toPath(), textFileEncoding);
      // Normalize line endings
      String text = lines.stream().collect(Collectors.joining("\n", "", "\n"));
      // Remove invalid characters
      String valid = invalidCharsPattern.matcher(text).replaceAll("");

      System.out.format(
          "Loaded and converted file: %d valid characters of %d total characters (%d removed)\n",
          valid.length(), text.length(), text.length() - valid.length());

      if (!valid.isEmpty()) {
        batch.add(valid);
        maxLength = Math.max(maxLength, valid.length());
      }
    }

    int batchSize = batch.size();
    // TODO: check empty?

    // We have batchSize examples of varying lengths
    INDArray features = Nd4j.create(new int[]{batchSize, validCharacters.length, maxLength}, 'f');
    INDArray labels = Nd4j.create(new int[]{batchSize, validCharacters.length, maxLength}, 'f');
    INDArray featuresMask = Nd4j.zeros(batchSize, maxLength, 'f');
    INDArray labelsMask = Nd4j.zeros(batchSize, maxLength, 'f');

    for (int i = 0; i < batchSize; i++) {
      String text = batch.get(i);
      int j = 0;
      // Current input
      int currCharIdx = charToIdxMap.get(text.charAt(j));
      for (int jEnd = text.length() - 1; j < jEnd; j++) {
        // Next character to predict
        int nextCharIdx = charToIdxMap.get(text.charAt(j + 1));
        features.putScalar(new int[]{i, currCharIdx, j}, 1.0);
        labels.putScalar(new int[]{i, nextCharIdx, j}, 1.0);
        featuresMask.putScalar(new int[]{i, j}, 1.0);
        labelsMask.putScalar(new int[]{i, j}, 1.0);
        currCharIdx = nextCharIdx;
      }
    }

    return new DataSet(features, labels, featuresMask, labelsMask);
  }

  @Override
  public int totalExamples() {
    return files.size();
  }

  @Override
  public int inputColumns() {
    return validCharacters.length;
  }

  @Override
  public int totalOutcomes() {
    return validCharacters.length;
  }

  @Override
  public final void reset() {
    Collections.shuffle(files, rng);
    cursor = 0;
  }

  @Override
  public boolean resetSupported() {
    return true;
  }

  @Override
  public boolean asyncSupported() {
    return true;
  }

  @Override
  public int batch() {
    return miniBatchSize;
  }

  @Override
  public int cursor() {
    return cursor;
  }

  @Override
  public int numExamples() {
    return totalExamples();
  }

  @Override
  public void setPreProcessor(DataSetPreProcessor preProcessor) {
    throw new UnsupportedOperationException("Not implemented");
  }

  @Override
  public DataSetPreProcessor getPreProcessor() {
    throw new UnsupportedOperationException("Not implemented");
  }

  @Override
  public List<String> getLabels() {
    throw new UnsupportedOperationException("Not implemented");
  }

  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }
}
