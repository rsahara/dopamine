//
//  SoftmaxWithCEELayer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/01.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

// TODO: generalize

// Softmax with cross entropy error as loss function.
public class SoftmaxWithCEELayer: TerminalLayer {

	public init() {
		_lastOutput = FloatBuffer(1, 1024)
		_lastOutputTarget = FloatBuffer(1, 1024)
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

	// MARK: - Hidden
	
	private var _lastOutput: FloatBuffer
	private var _lastTrainLoss: Float
	private var _lastOutputTarget: FloatBuffer
	
}
