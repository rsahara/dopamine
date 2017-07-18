//
//  ReluLayer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/01.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

class ReluLayer :Layer {
	
	override init() {
		_mask = FloatBuffer(1, 1024)
	}

	override func forward(input: FloatBuffer, result: FloatBuffer, forTraining: Bool) {

		_mask.resetLazy(like: input)
		result.resetLazy(like: input)
		
		if forTraining && hasPreviousLayer {
			_Layer_ResetZeroOrNegativeAndMakeMask(result.contents, _mask.contents, input.contents, Int32(input.capacity))
		}
		else {
			_Layer_ResetZeroOrNegative(result.contents, input.contents, Int32(input.capacity))
		}
	}
	
	override func backward(doutput: FloatBuffer, result: FloatBuffer) {

		if !hasPreviousLayer {
			return
		}
		
		result.copy(doutput)
		_Layer_ApplyMask(result.contents, _mask.contents, Int32(result.capacity));
	}

	private var _mask: FloatBuffer
	
}
