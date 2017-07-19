//
//  OptimizerRmsProp.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/02.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class OptimizerRmsProp : Optimizer {
	
	public init(learnRate: Float = 0.001, decayRate: Float = 0.9) {
		_learnRate = learnRate
		_decayRate = decayRate
		_tempBuffer1 = FloatBuffer(1, 1024 * 1024)
		_tempBuffer2 = FloatBuffer(1, 1024 * 1024)
		_iterationNum = 0
	}
	
	public func initialize(context: inout AnyObject?, rows: Int, columns: Int) {
		let h = FloatBuffer(rows, columns)
		h.fillZero()
		context = h as AnyObject
	}
	
	public func release(context: inout AnyObject?) {
		context = nil
	}
	
	public func updateIteration() {
		_iterationNum += 1
	}
	
	public func optimize(input: FloatBuffer, gradient: FloatBuffer, context: inout AnyObject?) { // TODO: optimize
		
		let h = context as! FloatBuffer

		assert(h._rows == gradient._rows)
		assert(h._columns == gradient._columns)

		h.mul(_decayRate)
		
		_tempBuffer1.copy(gradient)
		_tempBuffer1.mul(gradient)
		_tempBuffer1.mul(1.0 - _decayRate)
		h.add(_tempBuffer1)
		
		_tempBuffer2.copy(h)
		_tempBuffer2.sqrt()
		_tempBuffer2.add(0.000001)
		
		_tempBuffer1.copy(gradient)
		_tempBuffer1.mul(_learnRate)
		_tempBuffer1.div(_tempBuffer2)

		input.sub(_tempBuffer1)
	}
	
	// MARK: - Hidden
	
	private var _iterationNum: Int
	private let _learnRate: Float
	private let _decayRate: Float
	private var _tempBuffer1: FloatBuffer
	private var _tempBuffer2: FloatBuffer

}
