//
//  OptimizerDescent.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/02.
//  Copyright © 2017 Runo Sahara. All rights reserved.
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
	
	// MARK: - Hidden
	
	private let _learnRate: Float
	private var _tempBuffer: FloatBuffer
}
