//
//  Layer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/01.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public protocol Layer {

	func forwardPredict(input: FloatBuffer, result: FloatBuffer)
	func forwardTrain(input: FloatBuffer, result: FloatBuffer, hasPreviousLayer: Bool)
	func backwardTrain(dOutput: FloatBuffer, result: FloatBuffer, hasPreviousLayer: Bool)
	
	func initOptimizer(optimizer: Optimizer)
	func optimize(optimizer: Optimizer)
	
}

public protocol TerminalLayer {

	func forwardPredict(input: FloatBuffer, result: FloatBuffer)
	func forwardTrain(input: FloatBuffer, outputTarget: FloatBuffer)
	func backwardTrain(result: FloatBuffer)

}
