//
//  ReluLayer.swift
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/01.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

class ReluLayer :SimpleLayer {
	
	override init() {
		mask = FloatBuffer(1, 1024)
	}

	// (1, n) -> (1, n)
	override func forward(input: FloatBuffer, result: FloatBuffer, forTraining: Bool) {

		mask.resetLazy(like: input)
		result.resetLazy(like: input)
		
		if forTraining && hasPreviousLayer {
			input.maskZeroOrNegative(result: result, mask: mask)
		}
		else {
			input.resetZeroOrNegative(result: result)
		}
	}
	
	// (1, n) -> (1, n)
	override func backward(doutput: FloatBuffer, result: FloatBuffer) {

		if !hasPreviousLayer {
			return
		}
		
		result.copy(doutput)
		result.applyMask(mask)
	}

	var mask: FloatBuffer
	
}
