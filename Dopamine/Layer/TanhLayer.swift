//
//  TanhLayer.swift
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/11.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation


class TanhLayer: SimpleLayer {
	
	override init() {
		
		_lastOutput = FloatBuffer(1, 1024 * 1024)
		
	}
	
	override func forward(input: FloatBuffer, result: FloatBuffer, forTraining: Bool) {

		result.resetLazy(like: input)
		Layer_Tanh(result.contents, input.contents, Int32(input.capacity))
		
		if forTraining && hasPreviousLayer {
			_lastOutput.copy(result)
		}
		
	}
	
	override func backward(doutput: FloatBuffer, result: FloatBuffer) {
		
		if !hasPreviousLayer {
			return
		}
		
		assert(doutput.capacity == lastOutput.capacity)
		
		result.resetLazy(like: doutput)
		Layer_TanhBackward(result.contents, doutput.contents, lastOutput.contents, Int32(doutput.capacity));
	}
	
	var lastOutput: FloatBuffer {
		return _lastOutput
	}
	
	private var _lastOutput: FloatBuffer
	
}
