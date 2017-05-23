//
//  Layer.swift
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/01.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

class SimpleLayer {
	
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
	
	var hasPreviousLayer: Bool
}
