//
//  LayerNet.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/04/28.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

// TODO: generalize interface
// TODO: optimize

public class LayerNet {
	
	public init(inputSize: Int, hiddenSize: Int, outputSize: Int) {
		
		self.inputSize = inputSize
		self.hiddenSize = hiddenSize
		self.outputSize = outputSize

		layers = Array()
		layers.append(AffineLayer(inputSize: inputSize, outputSize: 50, layerName: "^", debugLog: false))
		layers.append(ReluLayer())
//		layers.append(AffineLayer(inputSize: 100, outputSize: 50, layerName: ":"))
//		layers.append(ReluLayer())
		layers.append(AffineLayer(inputSize: 50, outputSize: outputSize, layerName: "$", debugLog: false))
		
		tempBuffer1 = FloatBuffer(1, 1024 * 1024)
		tempBuffer2 = FloatBuffer(1, 1024 * 1024)
		
		lastLayer = SoftmaxWithLoss()
		
//		optimizer = OptimizerDescent(learnRate: 0.1)
//		optimizer = OptimizerAdam()
		optimizer = OptimizerRmsProp()

		for layer in layers {
			layer.initOptimizer(optimizer: optimizer)
		}
		layers.first!.hasPreviousLayer = false
	}
	
	//
	public func predict(input: FloatBuffer, result: FloatBuffer) {

		var nextInput = tempBuffer1
		var nextResult = tempBuffer2

		let firstLayer = layers.first!
		firstLayer.forward(input: input, result: nextResult, forTraining: false)

		for layerIndex in 1 ..< layers.count {
			let temp = nextInput
			nextInput = nextResult
			nextResult = temp

			let layer = layers[layerIndex]
			layer.forward(input: nextInput, result: nextResult, forTraining: false)
		}
		
		lastLayer.forward(input: nextResult, output: result, forTraining: false)
	}

	//
	public func train(input: FloatBuffer, output: FloatBuffer) {

		var nextInput = tempBuffer1
		var nextResult = tempBuffer2
		
		let firstLayer = layers.first!
		firstLayer.forward(input: input, result: nextResult, forTraining: true)

		for layerIndex in 1 ..< layers.count {
			let temp = nextInput
			nextInput = nextResult
			nextResult = temp

			let layer = layers[layerIndex]
			layer.forward(input: nextInput, result: nextResult, forTraining: true)
		}

		lastLayer.forward(input: nextResult, output: output, forTraining: true)
		
		optimizer.updateIteration()

		lastLayer.backward(result: nextResult)
		for layerIndex in stride(from: layers.count - 1, to: -1, by: -1) {
			let layer = layers[layerIndex]
			
			let temp = nextInput
			nextInput = nextResult
			nextResult = temp

			layer.backward(doutput: nextInput, result: nextResult)
			layer.optimize(optimizer: self.optimizer)
		}
	}

	let optimizer: Optimizer
	let inputSize: Int
	let hiddenSize: Int
	let outputSize: Int
	
	var layers: Array<Layer>
	var lastLayer: SoftmaxWithLoss
	
	var tempBuffer1: FloatBuffer
	var tempBuffer2: FloatBuffer

}
