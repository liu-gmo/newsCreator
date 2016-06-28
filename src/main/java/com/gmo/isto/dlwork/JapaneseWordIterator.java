package com.gmo.isto.dlwork;

import com.gmo.isto.dlwork.tools.DocItem;
import com.gmo.isto.dlwork.tools.JapanesePreProcess;
import com.gmo.isto.dlwork.tools.JapaneseTokenizer;
import com.gmo.isto.dlwork.tools.JapaneseTokenizerFactory;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created by usr0101862 on 2016/06/22.
 */
public class JapaneseWordIterator implements DataSetIterator {
    public static final int MinWordFreq = 10;

    private Map<String, Integer> wordFreqMap;
    //Maps each character to an index ind the input/output
    private Map<String,Integer> wordToIdxMap;
    private List idxToWord;
    //All characters of the input file (after filtering to only those that are valid
    //private List<dWord> docWords;

    private List words;

    //Length of each example/minibatch (number of words)
    private int exampleLength;
    //Size of each minibatch (number of examples)
    private int miniBatchSize;
    private Random rng;
    protected AtomicInteger position = new AtomicInteger(0);
    //Offsets for the start of each example
    private LinkedList<Integer> exampleStartOffsets = new LinkedList<Integer>();

    class dWord{
        private int docIndex;
        private String word;

        public dWord(int id, String w){
            this.docIndex = id;
            this.word = w;
        }

        public int getDocIndex() {
            return docIndex;
        }

        public void setDocIndex(int docIndex) {
            this.docIndex = docIndex;
        }

        public String getWord() {
            return word;
        }

        public void setWord(String word) {
            this.word = word;
        }
    }

    /**
     * @param textFilePath Path to text file to use for generating samples
     * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
     * @param miniBatchSize Number of examples per mini-batch
     * @param exampleLength Number of characters in each input/output vector
     * @param validCharacters Character array of valid characters. Characters not present in this array will be removed
     * @param rng Random number generator, for repeatability if required
     * @throws IOException If text file cannot  be loaded
     */
    public JapaneseWordIterator(List<DocItem> docs, int miniBatchSize, int exampleLength, Random rng) throws IOException {
        if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
        this.exampleLength = exampleLength;
        this.miniBatchSize = miniBatchSize;
        this.rng = rng;

        TokenizerFactory t = new JapaneseTokenizerFactory();
        //t.setTokenPreProcessor(new JapanesePreProcess());

        //Store valid characters is a map for later use in vectorization
        wordToIdxMap = new HashMap<String,Integer>();
        wordFreqMap = new HashMap<>();
        //docWords = new ArrayList<dWord>();
        words = new ArrayList<String>();

        int dCnt = 1;
        int wMax = 0;
        for(DocItem doc : docs){
            //String id = doc.getDocId();
            String content = doc.getDocContent();

            JapaneseTokenizer tz = (JapaneseTokenizer) t.create(content);

            while(tz.hasMoreTokens()) {
                String word = tz.nextToken();
                if(word != null){
                    words.add(word);
                    wMax++;

                    Integer freq = 1;
                    if(!wordFreqMap.containsKey(word)){
                        wordFreqMap.put(word, freq);
                    }
                    else{
                        freq = wordFreqMap.get(word) + 1;
                        wordFreqMap.put(word,  freq);
                    }

                    if(freq > MinWordFreq){ // add frequent words only
                        if(!wordToIdxMap.containsKey(word))
                            wordToIdxMap.put(word, position.getAndIncrement());
                    }
                }
            }

            dCnt++;
        }

        idxToWord = new ArrayList(wordToIdxMap.keySet());

        Iterator it = words.iterator();
        int wCnt = 0;
        while (it.hasNext()) {
            String word =  (String) it.next();
            if(!wordToIdxMap.containsKey(word)){//remove infrequent words
                it.remove();
                continue;
            }

            wCnt++;
        }
        //words = new ArrayList(docWords.values());

        if( exampleLength >= wCnt ) throw new IllegalArgumentException("exampleLength="+exampleLength
                +" cannot exceed number of valid characters in file ("+wCnt+")");

        int nRemoved = wMax - wCnt;
        System.out.println("Loaded and converted file: " + wCnt + " valid characters of "
                + wMax + " total characters (" + nRemoved + " removed)");

        //This defines the order in which parts of the file are fetched
        int nMinibatchesPerEpoch = (wCnt-1) / exampleLength - 2;   //-2: for end index, and for partial example
        for( int i=0; i<nMinibatchesPerEpoch; i++ ){
            exampleStartOffsets.add(i * exampleLength);
        }
        Collections.shuffle(exampleStartOffsets,rng);
    }

    public String convertIndexToWord( int idx ){
         return (String) idxToWord.get(idx);
    }

    public int convertWordToIndex( String c ){
        return wordToIdxMap.get(c);
    }

    public String getRandomWord(){
        return (String)idxToWord.get((int) (rng.nextDouble()*idxToWord.size()));
    }

    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    public DataSet next() {
        return next(miniBatchSize);
    }

    public DataSet next(int num) {
        if( exampleStartOffsets.size() == 0 ) throw new NoSuchElementException();

        int currMinibatchSize = Math.min(num, exampleStartOffsets.size());
        //Allocate space:
        //Note the order here:
        // dimension 0 = number of examples in minibatch
        // dimension 1 = size of each vector (i.e., number of characters)
        // dimension 2 = length of each time series/example
        INDArray input = Nd4j.zeros(currMinibatchSize,idxToWord.size(),exampleLength);
        INDArray labels = Nd4j.zeros(currMinibatchSize,idxToWord.size(),exampleLength);

        for( int i=0; i<currMinibatchSize; i++ ){
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            int currCharIdx = wordToIdxMap.get(words.get(startIdx));	//Current input
            int c=0;
            for( int j=startIdx+1; j<endIdx; j++, c++ ){
                int nextCharIdx = wordToIdxMap.get(words.get(j));		//Next character to predict
                input.putScalar(new int[]{i,currCharIdx,c}, 1.0);
                labels.putScalar(new int[]{i,nextCharIdx,c}, 1.0);
                currCharIdx = nextCharIdx;
            }
        }

        return new DataSet(input,labels);
    }

    public int totalExamples() {
        return (words.size()-1) / miniBatchSize - 2;
    }

    public int inputColumns() {
        return idxToWord.size();
    }

    public int totalOutcomes() {
        return idxToWord.size();
    }

    public void reset() {
        exampleStartOffsets.clear();
        int nMinibatchesPerEpoch = totalExamples();
        for( int i=0; i<nMinibatchesPerEpoch; i++ ){
            exampleStartOffsets.add(i * miniBatchSize);
        }
        Collections.shuffle(exampleStartOffsets,rng);
    }

    public int batch() {
        return miniBatchSize;
    }

    public int cursor() {
        return totalExamples() - exampleStartOffsets.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor preProcessor) {
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
