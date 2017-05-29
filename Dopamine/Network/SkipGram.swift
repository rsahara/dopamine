//
//  SkipGram.swift
//  Pods
//
//  Created by 佐原 瑠能 on 2017/05/26.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

class SkipGram {
	
	init(itemCapacity: Int, itemVectorSize: Int) {

		self.itemVectorSize = itemVectorSize
		self.weight = FloatBuffer(itemCapacity, itemVectorSize)
		self.weight.fillRandom()
		self.weightNeg = FloatBuffer(itemCapacity, itemVectorSize)
		self.weightNeg.fillZero()
		self.itemSelectBuffer = FloatBuffer(1, itemCapacity)
		self.itemCount = 0
		
	}

	func trainWithSequences(itemSequenceArray: [[Int]]) {
		// TODO: C++化

		// 数をカウントする
		self.itemSelectBuffer.fillZero()
		for itemSequence in itemSequenceArray {
			for itemIndex in itemSequence {
				itemSelectBuffer[itemIndex] += 1.0
				if (itemIndex > itemCount) {
					itemCount = itemIndex + 1
				}
			}
		}

		// ランダム選択の処理を準備
		var itemSelectSum: Float = 0.0
		for itemIndex in 0 ..< itemCount {
			itemSelectBuffer[itemIndex] = powf(itemSelectBuffer.contents[itemIndex], 0.75)
			itemSelectSum += itemSelectBuffer.contents[itemIndex];
		}
		let itemSelectSumInv: Float = 1.0 / itemSelectSum
		var itemSelectStep: Float = 0.0
		for itemIndex in 0 ..< itemCount {
			itemSelectStep += itemSelectBuffer.contents[itemIndex] * itemSelectSumInv
		}
		
		// 学習
		let neu1e = FloatBuffer(1, itemVectorSize)
		for itemSequence in itemSequenceArray {
			for itemIndex in itemSequence {
				
				for relatedItemIndex in itemSequence {
					
					if itemIndex == relatedItemIndex {
						continue
					}
					
					neu1e.fillZero()
					let headRelated = weight.contents + (itemVectorSize * relatedItemIndex)
					
					// Negative Sample
					var targetIndex: Int = 0
					var label: Float = 0
					for negativeIndex in 0 ..< negativeCount {
						
						if negativeIndex == 0 {
							targetIndex = itemIndex
							label = 1.0
						} else {
							targetIndex = selectRandomNegativeItemIndex()
							if targetIndex == itemIndex {
								continue
							}
							label = 0.0
						}
						
						let headTarget = weightNeg.contents + (itemVectorSize * targetIndex)
						var dot: Float = 0.0
						for featureIndex in 0 ..< itemVectorSize {
							dot += headRelated[featureIndex] * headTarget[featureIndex]
						}
						
						let expDot = expf(dot)
						let grad = label - (expDot / (expDot + 1.0))

						
					}

				}

				
				/*
				for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
					c = sentence_position - window + a;
					if (c < 0) continue;
					if (c >= sentence_length) continue;
					last_word = sen[c];
					if (last_word == -1) continue;
					l1 = last_word * layer1_size;
					for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
					
					
					// NEGATIVE SAMPLING
					if (negative > 0) for (d = 0; d < negative + 1; d++) {
						if (d == 0) {
							target = word;
							label = 1;
						} else {
							next_random = next_random * (unsigned long long)25214903917 + 11;
							target = table[(next_random >> 16) % table_size];
							if (target == 0) target = next_random % (vocab_size - 1) + 1;
							if (target == word) continue;
							label = 0;
						}
						l2 = target * layer1_size;
						f = 0;
						for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
						if (f > MAX_EXP) g = (label - 1) * alpha;
						else if (f < -MAX_EXP) g = (label - 0) * alpha;
						else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
						for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
						for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
					}
					// Learn weights input -> hidden
					for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
				}
				*/
				
				
			}
		}
	}

	func selectRandomNegativeItemIndex() -> Int {
		// TODO: テスト
		let randMax = itemSelectBuffer.contents[itemCount - 1]
		let randVal = Float(arc4random()) * (1.0 / Float(UINT32_MAX)) * randMax
		
		var minIndex = 0
		var maxIndex = itemCount - 1
		while minIndex < maxIndex {
			let testIndex = (minIndex + maxIndex) / 2
			let testVal = itemSelectBuffer.contents[testIndex]
			
			if randVal < testVal {
				maxIndex = testIndex - 1
			} else {
				minIndex = testIndex + 1
			}
		}
		return maxIndex % itemCount
	}

	struct ItemData {
		public var itemIndex: Int
		public var occurences: Int
	}
	
	let itemVectorSize: Int
	var itemCount: Int
	var weight: FloatBuffer
	var weightNeg: FloatBuffer
	let negativeCount: Int = 5
	
	var itemSelectBuffer: FloatBuffer
}
