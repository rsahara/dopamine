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
		
		_inputSize = inputSize
		_outputSize = outputSize
		_batchCapacity = batchCapacity
		
		_tempBuffer1 = FloatBuffer(1, 1024 * 1024) // TODO: generalize
		_tempBuffer2 = FloatBuffer(1, 1024 * 1024) // TODO: generalize
		
		_lastLayer = SoftmaxWithLoss() // TODO: generalize
		_optimizer = optimizer
		_layers = []
	}
	
	public func setup(layers: [Layer]) {
		_layers = layers

		for layer in _layers {
			layer.initOptimizer(optimizer: _optimizer)
		}
		
		_layers.first!.hasPreviousLayer = false
	}
	
	//
	public func predict(input: FloatBuffer, result: FloatBuffer) {

		var nextInput = _tempBuffer1
		var nextResult = _tempBuffer2

		let firstLayer = _layers.first!
		firstLayer.forward(input: input, result: nextResult, forTraining: false)

		for layerIndex in 1 ..< _layers.count {
			let temp = nextInput
			nextInput = nextResult
			nextResult = temp

			let layer = _layers[layerIndex]
			layer.forward(input: nextInput, result: nextResult, forTraining: false)
		}
		
		_lastLayer.forward(input: nextResult, output: result, forTraining: false)
	}

	//
	public func train(input: FloatBuffer, output: FloatBuffer) {

		var nextInput = _tempBuffer1
		var nextResult = _tempBuffer2
		
		let firstLayer = _layers.first!
		firstLayer.forward(input: input, result: nextResult, forTraining: true)

		for layerIndex in 1 ..< _layers.count {
			let temp = nextInput
			nextInput = nextResult
			nextResult = temp

			let layer = _layers[layerIndex]
			layer.forward(input: nextInput, result: nextResult, forTraining: true)
		}

		_lastLayer.forward(input: nextResult, output: output, forTraining: true)
		
		_optimizer.updateIteration()

		_lastLayer.backward(result: nextResult)
		for layerIndex in stride(from: _layers.count - 1, to: -1, by: -1) {
			let layer = _layers[layerIndex]
			
			let temp = nextInput
			nextInput = nextResult
			nextResult = temp

			layer.backward(doutput: nextInput, result: nextResult)
			layer.optimize(optimizer: _optimizer)
		}
	}
	
	// MARK: Hidden

	private let _inputSize: Int
	private let _outputSize: Int
	private let _batchCapacity: Int
	
	private var _layers: [Layer]
	private var _lastLayer: SoftmaxWithLoss
	private var _optimizer: Optimizer
	
	private var _tempBuffer1: FloatBuffer
	private var _tempBuffer2: FloatBuffer

}
