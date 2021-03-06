//
//  SoftmaxWithCEELayer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/01.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

// Softmax with cross entropy error as loss function.
public class SoftmaxWithCEELayer: TerminalLayer {

	public init(inputSize: Int, batchCapacity: Int) {
		_inputSize = inputSize
		_batchCapacity = batchCapacity
		_lastOutput = FloatBuffer(batchCapacity, inputSize)
		_lastOutputTarget = FloatBuffer(batchCapacity, inputSize)
		_lastTrainLoss = 0.0
	}

	public func forwardPredict(input: FloatBuffer, result: FloatBuffer) {
		input.softmax(result: result)
	}
	
	public func forwardTrain(input: FloatBuffer, outputTarget: FloatBuffer) {
		input.softmax(result: _lastOutput)
		_lastTrainLoss = _lastOutput.crossEntropyError(against: outputTarget)
		_lastOutputTarget.copy(from: outputTarget)
	}
	
	public func backwardTrain(result: FloatBuffer) {
		result.copy(from: _lastOutput)
		result.sub(_lastOutputTarget)
		result.mul(1.0 / Float(_lastOutputTarget.rows))
	}
	
	public func requiredResultCapacity() -> Int {
		return _inputSize * _batchCapacity
	}

	// MARK: - Hidden
	private let _inputSize: Int
	private let _batchCapacity: Int
	private var _lastOutput: FloatBuffer
	private var _lastTrainLoss: Float
	private var _lastOutputTarget: FloatBuffer
	
}
