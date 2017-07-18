//
//  Layer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/01.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

class Layer {
	
	init () {
		hasPreviousLayer = true
	}
	
	func forward(input: FloatBuffer, result: FloatBuffer, forTraining: Bool) {
	}

	func backward(doutput: FloatBuffer, result: FloatBuffer) {
	}

	func initOptimizer(optimizer: Optimizer) {
	}

	func optimize(optimizer: Optimizer){
	}
	
	internal var hasPreviousLayer: Bool
}
