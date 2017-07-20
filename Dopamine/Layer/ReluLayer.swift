//
//  ReluLayer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/01.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

// RELU layer.
public class ReluLayer: Layer {
	
	public init(inputSize: Int, batchCapacity: Int) {
		_mask = FloatBuffer(batchCapacity, inputSize)
	}

	public func forwardPredict(input: FloatBuffer, result: FloatBuffer) {
		_mask.reshape(like: input)
		result.reshape(like: input)

		_Layer_ResetZeroOrNegative(result.contents, input.contents, Int32(input.capacity))
	}
	
	public func forwardTrain(input: FloatBuffer, result: FloatBuffer, hasPreviousLayer: Bool) {
		_mask.reshape(like: input)
		result.reshape(like: input)

		if hasPreviousLayer {
			_Layer_ResetZeroOrNegativeAndMakeMask(result.contents, _mask.contents, input.contents, Int32(input.capacity))
		}
	}

	public func backwardTrain(dOutput: FloatBuffer, result: FloatBuffer, hasPreviousLayer: Bool) {
		
		if hasPreviousLayer {
			result.copy(dOutput)
			_Layer_ApplyMask(result.contents, _mask.contents, Int32(result.capacity));
		}
		
	}

	public func initOptimizer(optimizer: Optimizer) {
	}

	public func optimize(optimizer: Optimizer) {
	}

	// MARK: - Hidden
	
	private var _mask: FloatBuffer
	
}
