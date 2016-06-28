/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package com.gmo.isto.dlwork.tools;

import org.atilika.kuromoji.Token;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Default tokenizer
 * @author Guangwen Liu
 */
public class JapaneseTokenizer implements Tokenizer {
	List<Token> tokens = null;
	protected AtomicInteger position = new AtomicInteger(0);

	public JapaneseTokenizer(String text) {
		tokens = tokenizer.tokenize(text);
	}
	
	private static org.atilika.kuromoji.Tokenizer tokenizer = org.atilika.kuromoji.Tokenizer.builder()
			.mode(org.atilika.kuromoji.Tokenizer.Mode.NORMAL)
			.split(true)
			.build();

	private TokenPreProcess tokenPreProcess;
	
	@Override
	public boolean hasMoreTokens() {
		return position.get() < tokens.size();
	}

	@Override
	public int countTokens() {
		return tokens.size();
	}

	private boolean isBlackToken(Token token){
		String pos = token.getPartOfSpeech();

		if(pos.startsWith("記号") || pos.startsWith("助詞") || pos.startsWith("助動詞"))
			return true;
		else
			return  false;
	}

	private boolean isWhiteToken(Token token){
		String pos = token.getPartOfSpeech();

		if(pos.startsWith("名詞") || pos.startsWith("動詞") || pos.startsWith("形容詞"))
			return true;
		else
			return  false;
	}

	@Override
	public String nextToken() {
		if(!hasMoreTokens()) return null;

		Token token = tokens.get(position.getAndIncrement());
//		if(isBlackToken(token)){
//			return nextToken();
//		}
		String base = token.getSurfaceForm();
		if(base.length() > 30) {
			System.out.println("too long: " + base);
			return nextToken();
		}
		if(tokenPreProcess != null)
            base = tokenPreProcess.preProcess(base);
        return base;
	}

	@Override
	public List<String> getTokens() {
		List<String> tokens = new ArrayList<>();
		while(hasMoreTokens()) {
			String tkn = nextToken();
			if(tkn != null)
				tokens.add(tkn);
		}
		return tokens;
	}

	@Override
	public void setTokenPreProcessor(TokenPreProcess tokenPreProcessor) {
		this.tokenPreProcess = tokenPreProcessor;
		
	}
}
