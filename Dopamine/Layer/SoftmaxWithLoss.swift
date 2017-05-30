//
//  SoftmaxWithLoss.swift
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/01.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

class SoftmaxWithLoss {

	init() {
		_lastOutput = FloatBuffer(1, 1024)
		_lastTrainOutput = FloatBuffer(1, 1024)
		_lastTrainLoss = 0.0
	}
	
	func forward(input: FloatBuffer, output: FloatBuffer, forTraining: Bool) {

		input.softmax(result: _lastOutput)

		if forTraining {
			_lastTrainLoss = _lastOutput.crossEntropyError(against: output)
//			Swift.print("\(_lastTrainLoss)")
			_lastTrainOutput.copy(output)
		} else {
			output.copy(_lastOutput)
		}
	}
	
	// (1, n) -> (1, n)
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

	private var _lastOutput: FloatBuffer

	private var _lastTrainLoss: Float
	private var _lastTrainOutput: FloatBuffer
	
}
