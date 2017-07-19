//
//  OptimizerDescent.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/02.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class OptimizerDescent: Optimizer {

	public init(learnRate: Float = 0.01) {
		_learnRate = learnRate
		_tempBuffer = FloatBuffer(1, _initialCapacity)
	}

	public func initialize(context: inout AnyObject?, rows: Int, columns: Int) {
	}

	public func release(context: inout AnyObject?) {
	}

	public func updateIteration() {
	}
	
	public func optimize(input: FloatBuffer, gradient: FloatBuffer, context: inout AnyObject?) {
		
		// Reallocate if needed.
		if _tempBuffer.capacity < gradient.capacity {
			_tempBuffer = FloatBuffer(like: gradient)
		}
		
		_tempBuffer.copy(gradient)
		_tempBuffer.mul(_learnRate)
		input.sub(_tempBuffer)
	}
	
	// MARK: - Hidden
	
	private let _initialCapacity: Int = 1024 * 1024
	private let _learnRate: Float
	private var _tempBuffer: FloatBuffer
}
