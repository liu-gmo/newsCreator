package com.gmo.isto.dlwork.tools;

import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;

import java.util.regex.Pattern;

/**
 * Created by usr0101862 on 2016/06/26.
 */

public class JapanesePreProcess implements TokenPreProcess {
    private static final Pattern punctPattern = Pattern.compile("[\\s\\n]{2}");

    @Override
    public String preProcess(String token) {
        return stripPunct(token);
    }

    /**
     * Strip punctuation
     * @param base the base string
     * @return the cleaned string
     */
    public static String stripPunct(String base) {
        return punctPattern.matcher(base).replaceAll("");
    }
}
