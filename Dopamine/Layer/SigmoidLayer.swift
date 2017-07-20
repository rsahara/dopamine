//
//  SigmoidLayer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/11.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

// TODO: generalize
public class SigmoidLayer: Layer {

	public init(inputSize: Int, batchCapacity: Int) {
		_inputSize = inputSize
		_batchCapacity = batchCapacity
		_lastOutput = FloatBuffer(batchCapacity, _inputSize)
	}
	
	public func forwardPredict(input: FloatBuffer, result: FloatBuffer) {
		result.reshape(like: input)
		_Layer_Sigmoid(result.contents, input.contents, Int32(input.capacity))
	}
	
	public func forwardTrain(input: FloatBuffer, result: FloatBuffer, hasPreviousLayer: Bool) {
		forwardPredict(input: input, result: result)
		
		if hasPreviousLayer {
			_lastOutput.copy(from: result)
		}
	}
	
	public func backwardTrain(dOutput: FloatBuffer, result: FloatBuffer, hasPreviousLayer: Bool) {
		if hasPreviousLayer {
			assert(dOutput.capacity == _lastOutput.capacity)
			result.reshape(like: dOutput)
			_Layer_SigmoidBackward(result.contents, dOutput.contents, _lastOutput.contents, Int32(dOutput.capacity));
		}
	}
	
	public func requiredResultCapacity() -> Int {
		return _inputSize * _batchCapacity
	}

	public func initOptimizer(optimizer: Optimizer) {
	}
	
	public func optimize(optimizer: Optimizer) {
	}

	public var lastOutput: FloatBuffer { return _lastOutput }

	// MARK: - Hidden
	private let _inputSize: Int
	private let _batchCapacity: Int
	private let _lastOutput: FloatBuffer

}
