//
//  SkipGram.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/26.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class SkipGram {
	
	public static let EndOfItemSequenceId: Int32 = -1
	
	public init(itemCapacity: Int, itemVectorSize: Int, itemSequenceCapacity: Int = 1024, windowSize: Int = 100, negativeSamplingCount: Int = 5) {

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
		_windowSize = windowSize
		_negativeSamplingCount = negativeSamplingCount
	}

	public func train(itemSequenceBuffer: IntBuffer, itemSequenceLength: Int, iterationsCount: Int, trainingRate: Float = 0.005) {

		var itemSequencesCount = Int32(_itemSequenceCapacity)
		var itemsCount = Int32(_itemCapacity)
		
		_SkipGram_TrainInit(itemSequenceBuffer.contents,
		                    Int32(itemSequenceLength),
		                    _itemSequenceOffsetArray.contents,
		                    &itemSequencesCount,
		                    _itemNegLotteryInfoArray.contents,
		                    &itemsCount);
		
		_itemSequencesCount = Int(itemSequencesCount)
		_itemsCount = Int(itemsCount)
		
		_SkipGram_TrainIterate(itemSequenceBuffer.contents,
		                       _itemSequenceOffsetArray.contents,
		                       itemSequencesCount,
		                       _itemNegLotteryInfoArray.contents,
		                       itemsCount,
		                       Int32(_itemVectorSize),
							   _weight.contents,
							   _weightNeg.contents,
							   _tempItemVector.contents,
							   Int32(_windowSize),
							   Int32(_negativeSamplingCount),
							   trainingRate,
							   Int32(iterationsCount));
	}

	// MARK: - Properties
	
	public var result: FloatBuffer {
		return _weight;
	}

	// MARK: - Hidden

	private var _itemVectorSize: Int
	private var _itemCapacity: Int
	private var _itemSequenceCapacity: Int
	private var _itemsCount: Int
	private var _itemSequencesCount: Int
	private var _itemNegLotteryInfoArray: FloatBuffer
	private var _itemSequenceOffsetArray: IntBuffer
	private var _tempItemVector: FloatBuffer
	private var _weight: FloatBuffer
	private var _weightNeg: FloatBuffer
	private var _windowSize: Int
	private var _negativeSamplingCount: Int
}
