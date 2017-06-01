//
//  SkipGram.swift
//  Pods
//
//  Created by 佐原 瑠能 on 2017/05/26.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

public class SkipGram {
	
	public static let EndOfItemSequenceId: Int32 = -1
	
	public init(itemCapacity: Int, itemVectorSize: Int, itemSequenceCapacity: Int = 1024, negativeSamplingCount: Int = 5) {

		_SkipGram_GlobalInit()

		_itemCapacity = itemCapacity
		_itemVectorSize = itemVectorSize
		_itemSequenceCapacity = itemSequenceCapacity

		_weight = FloatBuffer(itemCapacity, itemVectorSize)
		_weight.fillRandom()
		_weight.mul(1.0 / Float(itemVectorSize))
		_weightNeg = FloatBuffer(itemCapacity, itemVectorSize)
		_weightNeg.fillZero()
		
		_itemNegLotteryInfoArray = FloatBuffer(1, itemCapacity)

		_itemsCount = 0
		_itemSequencesCount = 0
		_itemSequenceOffsetArray = IntBuffer(itemSequenceCapacity)
		_tempItemVector = FloatBuffer(1, itemVectorSize)
		_negativeSamplingCount = negativeSamplingCount
	}

	public func train(itemSequenceBuffer: IntBuffer, itemSequenceLength: Int) {

		// アイテムを処理する準備
		var itemSequencesCount: Int32 = Int32(_itemSequenceCapacity)
		var itemsCount: Int32 = Int32(_itemCapacity)
		
		_SkipGram_TrainInit(itemSequenceBuffer.contents, Int32(itemSequenceLength), _itemSequenceOffsetArray.contents, &itemSequencesCount, _itemNegLotteryInfoArray.contents, &itemsCount);
		
		_itemSequencesCount = Int(itemSequencesCount)
		_itemsCount = Int(itemsCount)
		
		let perfCheck = PerfCheck()
		for iterationIndex in 0 ..< 400 {
			Swift.print("it \(iterationIndex)")
			
			_SkipGram_TrainIterate(itemSequenceBuffer.contents, _itemSequenceOffsetArray.contents, itemSequencesCount, _itemNegLotteryInfoArray.contents, itemsCount, Int32(_itemVectorSize),
			                       _weight.contents, _weightNeg.contents, _tempItemVector.contents, 100, 5, 0.005);// param
		}
		perfCheck.print()

	}

	// MARK: - プロパティ
	
	var _itemVectorSize: Int
	var _itemCapacity: Int
	var _itemSequenceCapacity: Int
	var _itemsCount: Int
	var _itemSequencesCount: Int
	var _itemNegLotteryInfoArray: FloatBuffer
	var _itemSequenceOffsetArray: IntBuffer
	var _tempItemVector: FloatBuffer
	
	public var _weight: FloatBuffer
	var _weightNeg: FloatBuffer
	var _negativeSamplingCount: Int
}
