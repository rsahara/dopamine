//
//  OptimizerDescent.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/02.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class OptimizerDescent: Optimizer {
	
	let initialCapacity = 102400
	
	public init(learnRate: Float = 0.01) {
		_learnRate = learnRate
		_tempBuffer = FloatBuffer(1, initialCapacity)
	}

	public func initialize(context: inout AnyObject?, rows: Int, columns: Int) {
	}

	public func release(context: inout AnyObject?) {
	}

	public func updateIteration() {
	}
	
	public func optimize(input: FloatBuffer, gradient: FloatBuffer, context: inout AnyObject?) {
		_tempBuffer.reshape(like: gradient)
		_tempBuffer.copy(gradient)
		_tempBuffer.mul(_learnRate)
		input.sub(_tempBuffer)
	}
	
	// MARK: - Hidden
	
	private let _learnRate: Float
	private var _tempBuffer: FloatBuffer
}
