//
//  SoftmaxWithLoss.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/01.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class SoftmaxWithLoss {

	public init() {
		_lastOutput = FloatBuffer(1, 1024)
		_lastTrainOutput = FloatBuffer(1, 1024)
		_lastTrainLoss = 0.0
	}
	
	func forward(input: FloatBuffer, output: FloatBuffer, forTraining: Bool) {

		input.softmax(result: _lastOutput)

		if forTraining {
			_lastTrainLoss = _lastOutput.crossEntropyError(against: output)
			_lastTrainOutput.copy(output)
		} else {
			output.copy(_lastOutput)
		}
	}
	
	func backward(result: FloatBuffer) {
		result.copy(_lastOutput)
		result.sub(_lastTrainOutput)
		result.mul(1.0 / Float(_lastTrainOutput.rows))
	}
	
	public var lastOutput: FloatBuffer {
		return _lastOutput
	}
	
	public var lastTrainLoss: Float {
		return _lastTrainLoss
	}

	// MARK: - Hidden
	
	private var _lastOutput: FloatBuffer
	private var _lastTrainLoss: Float
	private var _lastTrainOutput: FloatBuffer
	
}
