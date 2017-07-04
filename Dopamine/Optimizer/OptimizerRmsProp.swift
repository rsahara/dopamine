//
//  OptimizerRmsProp.swift
//  RunoNetTest
//
//  Created by Runo Sahara on 2017/05/02.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

class OptimizerRmsProp : Optimizer {
	
	init(learnRate: Float = 0.001, decayRate: Float = 0.9) {
		_learnRate = learnRate
		_decayRate = decayRate
		_tempBuffer1 = FloatBuffer(1, 1024 * 1024)
		_tempBuffer2 = FloatBuffer(1, 1024 * 1024)
		_iterationNum = 0
	}
	
	func initialize(context: inout AnyObject?) {
		let h = FloatBuffer(1, 1024 * 1024)
		context = h as AnyObject
	}
	
	func release(context: inout AnyObject?) {
		context = nil
	}
	
	func updateIteration() {
		_iterationNum += 1
	}
	
	func optimize(input: FloatBuffer, gradient: FloatBuffer, context: inout AnyObject?) {
		
		let h = context as! FloatBuffer

		if _iterationNum == 1 {
			h.resetLazy(like: input) // TODO: リファクタ
			h.fillZero()
		}
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
	
	var _iterationNum: Int
	let _learnRate: Float
	let _decayRate: Float
	var _tempBuffer1: FloatBuffer
	var _tempBuffer2: FloatBuffer

}
