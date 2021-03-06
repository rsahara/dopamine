//
//  AdamOptimizer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/02.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

// experimental
#if false
	
	
import Foundation

// TODO: generalize/refactor
public class AdamOptimizer: Optimizer {
	
	public init(learnRate: Float = 0.001, beta1: Float = 0.9, beta2: Float = 0.999) {
		_learnRate = learnRate
		_beta1 = beta1
		_beta2 = beta2
		_iterationNum = 0
		_step = 0.0
	}

	public func initialize(context: inout AnyObject?) {
		let m = FloatBuffer(1, 1024 * 1024)
		let v = FloatBuffer(1, 1024 * 1024)
		
		context = [m, v] as AnyObject
	}
	
	public func release(context: inout AnyObject?) {
		context = nil
	}

	public func updateIteration() {
		_iterationNum += 1
		_step = _learnRate * sqrtf(1.0 - powf(_beta2, Float(_iterationNum))) / (1.0 - powf(_beta1, Float(_iterationNum))) // TODO: optimize
	}
	
	public func optimize(input: FloatBuffer, gradient: FloatBuffer, context: inout AnyObject?) { // TODO: optimize

		let contextArray = context as! Array<FloatBuffer>
		let m: FloatBuffer = contextArray[0]
		let v: FloatBuffer = contextArray[1]
		
		if (_iterationNum == 1) {
			m.resetLazy(like: input)
			v.resetLazy(like: input)
			m.fillZero()
			v.fillZero()
		}
		
		let temp = FloatBuffer(like: gradient)
		temp.copy(gradient)
		temp.sub(m)
		temp.mul(1.0 - _beta1)
		m.add(temp)
		
		temp.copy(gradient)
		temp.mul(gradient)
		temp.sub(v)
		temp.mul(1.0 - _beta2)
		v.add(temp)

		let temp2 = FloatBuffer(like: v)
		temp2.copy(v)
		temp2.sqrt()
		temp2.add(0.0000001)
		
		temp.copy(m)
		temp.mul(_step)
		temp.div(temp2)
		
		input.sub(temp)
	}
	
	// MARK: - Hidden
	
	private var _iterationNum: Int
	private var _step: Float
	private let _learnRate: Float
	private let _beta1: Float
	private let _beta2: Float
}

#endif
