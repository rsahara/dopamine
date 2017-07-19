//
//  ReluLayer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/01.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class ReluLayer: Layer {
	
	public init(inputSize: Int, batchCapacity: Int) {
		_mask = FloatBuffer(batchCapacity, inputSize)
		super.init()
	}

	override func forward(input: FloatBuffer, result: FloatBuffer, forTraining: Bool) {

		_mask.reshape(like: input)
		result.reshape(like: input)

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

	// MARK: - Hidden
	
	private var _mask: FloatBuffer
	
}
