//
//  OptimizerDescent.swift
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/02.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

class OptimizerDescent: Optimizer {
	
	init(learnRate: Float = 0.01) {
		_learnRate = learnRate
		_tempBuffer = FloatBuffer(100, 1024)
	}

	func initialize(context: inout AnyObject?) {
	}

	func release(context: inout AnyObject?) {
	}

	func updateIteration() {
	}
	
	func optimize(input: FloatBuffer, gradient: FloatBuffer, context: inout AnyObject?) {
		_tempBuffer.copy(gradient)
		_tempBuffer.mul(_learnRate)
		input.sub(_tempBuffer)
	}
	
	let _learnRate: Float
	var _tempBuffer: FloatBuffer
}
