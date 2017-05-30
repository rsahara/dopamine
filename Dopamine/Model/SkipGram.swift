//
//  SkipGram.swift
//  Pods
//
//  Created by 佐原 瑠能 on 2017/05/26.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

public class SkipGram {
	
	public init(itemCapacity: Int, itemVectorSize: Int) {

		self.itemVectorSize = itemVectorSize
		self.weight = FloatBuffer(itemCapacity, itemVectorSize)
		self.weight.fillRandom()
		self.weight.mul(1.0 / Float(itemVectorSize))
		self.weightNeg = FloatBuffer(itemCapacity, itemVectorSize)
		self.weightNeg.fillZero()
		self.itemSelectBuffer = FloatBuffer(1, itemCapacity)
		self.itemCount = 0
		
	}

	public func trainWithSequences(itemSequenceArray: [[Int]]) {
		
		// TODO: C++化

		// 数をカウントする
		self.itemSelectBuffer.fillZero()
		for itemSequence in itemSequenceArray {
			for itemIndex in itemSequence {
				itemSelectBuffer.contents[itemIndex] += 1.0
				if (itemIndex > itemCount) {
					itemCount = itemIndex + 1
				}
			}
		}

		// ランダム選択の処理を準備
		var itemSelectSum: Float = 0.0
		for itemIndex in 0 ..< itemCount {
			itemSelectBuffer.contents[itemIndex] = powf(itemSelectBuffer.contents[itemIndex], 0.75)
			itemSelectSum += itemSelectBuffer.contents[itemIndex];
		}
		let itemSelectSumInv: Float = 1.0 / itemSelectSum
		var itemSelectStep: Float = 0.0
		for itemIndex in 0 ..< itemCount {
			itemSelectStep += itemSelectBuffer.contents[itemIndex] * itemSelectSumInv
			itemSelectBuffer.contents[itemIndex] = itemSelectStep
		}
		
		// 学習
		for iterationIndex in 0 ..< 400 {
			Swift.print("it \(iterationIndex)")
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
						for negativeIndex in 0 ..< negativeCount + 1 {
							
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
							let grad = (label - (expDot / (expDot + 1.0))) * 0.001 // TODO: learningRate
							
							for featureIndex in 0 ..< itemVectorSize {
								neu1e.contents[featureIndex] += grad * headTarget[featureIndex]
							}
							for featureIndex in 0 ..< itemVectorSize {
								headTarget[featureIndex] += grad * headRelated[featureIndex]
							}
						}

						for featureIndex in 0 ..< itemVectorSize {
							headRelated[featureIndex] += neu1e.contents[featureIndex]
						}

					}
				}

			}
		}

		// テスト
//		weight.print()
	}

	func selectRandomNegativeItemIndex() -> Int {
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
	public var weight: FloatBuffer
	var weightNeg: FloatBuffer
	let negativeCount: Int = 5
	
	var itemSelectBuffer: FloatBuffer
}
