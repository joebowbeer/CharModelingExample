/*
 * Copyright 2017 Joe Bowbeer.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.joebowbeer.dl4j.recurrent.character;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.regex.Pattern;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

/**
 *
 */
public class DanceIterator implements DataSetIterator {
  private final Document doc;
  private final List<String> titles = new ArrayList<>();
//  private final Charset textFileEncoding;
  // Size of each minibatch (number of examples)
  private final int miniBatchSize;
  // Valid characters sorted for binary search
  private final char[] sortedChars;
  // Regex matching invalid characters
  private final Pattern invalidCharsPattern;
  private final Random rng;

  private int cursor;


  public DanceIterator(File file, Charset encoding, int miniBatchSize, char[] validCharacters,
      Random rng) throws IOException {
    if (!file.exists()) {
      throw new IOException("Could not access file (does not exist): " + file);
    }
    if (miniBatchSize <= 0) {
      throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
    }

    this.doc = Jsoup.parse(file, encoding.name(), "");
    for (Element e : doc.select("a[name]")) {
      String s = e.nextSibling().toString();
      String title = s.split("\n")[1];
      titles.add("^" + title + "\n");
    }
    System.out.println("Size=" + titles.size());
    System.out.println(titles);
    
    this.miniBatchSize = miniBatchSize;
    sortedChars = Arrays.copyOf(validCharacters, validCharacters.length);
    Arrays.sort(sortedChars);
    this.rng = rng;

    // Create regex that matches invalid characters
    String escaped = new String(sortedChars).replaceAll("[\\W]", "\\\\$0");
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
    return sortedChars[idx];
  }

  public int convertCharacterToIndex(char c) {
    return Arrays.binarySearch(sortedChars, c);
  }

  public char getRandomCharacter() {
    return sortedChars[rng.nextInt(sortedChars.length)];
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
      String text = titles.get(cursor);
      // Remove invalid characters
      String valid = invalidCharsPattern.matcher(text).replaceAll("");

//      System.out.format(
//          "Loaded and converted %s:\t %d valid characters of %d total characters (%d removed)\n",
//          file.getName(), valid.length(), text.length(), text.length() - valid.length());

      if (!valid.isEmpty()) {
        batch.add(valid);
        maxLength = Math.max(maxLength, valid.length());
      }
    }

    int batchSize = batch.size();
    // TODO: check empty?

    // We have batchSize examples of varying lengths
    INDArray features = Nd4j.create(new int[]{batchSize, sortedChars.length, maxLength}, 'f');
    INDArray labels = Nd4j.create(new int[]{batchSize, sortedChars.length, maxLength}, 'f');
    INDArray featuresMask = Nd4j.zeros(batchSize, maxLength, 'f');
    INDArray labelsMask = Nd4j.zeros(batchSize, maxLength, 'f');

    for (int i = 0; i < batchSize; i++) {
      String text = batch.get(i);
      // Current input
      int currCharIdx = convertCharacterToIndex(text.charAt(0));
      for (int j = 0; j < text.length() - 1; j++) {
        features.putScalar(new int[]{i, currCharIdx, j}, 1.0);
        // Next character to predict
        int nextCharIdx = convertCharacterToIndex(text.charAt(j + 1));
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
    return titles.size();
  }

  @Override
  public int inputColumns() {
    return sortedChars.length;
  }

  @Override
  public int totalOutcomes() {
    return sortedChars.length;
  }

  @Override
  public final void reset() {
    Collections.shuffle(titles, rng);
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
