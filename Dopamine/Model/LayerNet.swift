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
	
	public init(inputSize: Int, outputSize: Int, batchCapacity: Int, optimizer: Optimizer) {
		assert(inputSize > 0)
		assert(outputSize > 0)
		assert(batchCapacity > 0)
		
		self.inputSize = inputSize
		self.outputSize = outputSize
		_batchCapacity = batchCapacity
		
		tempBuffer1 = FloatBuffer(1, 1024 * 1024)
		tempBuffer2 = FloatBuffer(1, 1024 * 1024)
		
		lastLayer = SoftmaxWithLoss()
		self.optimizer = optimizer
		layers = []
	}
	
	public func setup(layers: [Layer]) {
		self.layers = layers

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

	let inputSize: Int
	let outputSize: Int
	let _batchCapacity: Int
	
	var layers: [Layer]
	var lastLayer: SoftmaxWithLoss
	var optimizer: Optimizer
	
	var tempBuffer1: FloatBuffer
	var tempBuffer2: FloatBuffer

}
