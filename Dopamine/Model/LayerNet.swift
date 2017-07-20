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
	
	public init(inputSize: Int, outputSize: Int, batchCapacity: Int, optimizer: Optimizer, terminalLayer: TerminalLayer) {
		assert(inputSize > 0)
		assert(outputSize > 0)
		assert(batchCapacity > 0)
		
		_inputSize = inputSize
		_outputSize = outputSize
		_batchCapacity = batchCapacity
		
		_tempBuffer1 = FloatBuffer(1, 1)
		_tempBuffer2 = FloatBuffer(1, 1)
		
		_terminalLayer = terminalLayer
		_optimizer = optimizer
		_layers = []
	}
	
	public func setup(layers: [Layer]) {
		_layers = layers

		var maxResultCapacity = _terminalLayer.requiredResultCapacity()
		for layer in _layers {
			layer.initOptimizer(optimizer: _optimizer)
			
			let requiredResultCapacity = layer.requiredResultCapacity()
			if requiredResultCapacity > maxResultCapacity {
				maxResultCapacity = requiredResultCapacity
			}
		}
		
		_tempBuffer1 = FloatBuffer(1, maxResultCapacity)
		_tempBuffer2 = FloatBuffer(1, maxResultCapacity)
	}

	public func predict(input: FloatBuffer, result: FloatBuffer) {

		var nextInput = _tempBuffer1
		var nextResult = _tempBuffer2

		let firstLayer = _layers.first!
		firstLayer.forwardPredict(input: input, result: nextResult)

		for layerIndex in 1 ..< _layers.count {
			let temp = nextInput
			nextInput = nextResult
			nextResult = temp

			let layer = _layers[layerIndex]
			layer.forwardPredict(input: nextInput, result: nextResult)
		}
		
		_terminalLayer.forwardPredict(input: nextResult, result: result)
	}

	public func train(input: FloatBuffer, outputTarget: FloatBuffer) {

		var nextInput = _tempBuffer1
		var nextResult = _tempBuffer2
		
		let firstLayer = _layers.first!
		firstLayer.forwardTrain(input: input, result: nextResult, hasPreviousLayer: false)

		for layerIndex in 1 ..< _layers.count {
			let temp = nextInput
			nextInput = nextResult
			nextResult = temp

			let layer = _layers[layerIndex]
			layer.forwardTrain(input: nextInput, result: nextResult, hasPreviousLayer: true)
		}

		_terminalLayer.forwardTrain(input: nextResult, outputTarget: outputTarget)
		
		_optimizer.updateIteration()

		_terminalLayer.backwardTrain(result: nextResult)
		for layerIndex in stride(from: _layers.count - 1, to: -1, by: -1) {
			let layer = _layers[layerIndex]
			
			let temp = nextInput
			nextInput = nextResult
			nextResult = temp

			layer.backwardTrain(dOutput: nextInput, result: nextResult, hasPreviousLayer: true)
			layer.optimize(optimizer: _optimizer)
		}
	}
	
	// MARK: Hidden

	private let _inputSize: Int
	private let _outputSize: Int
	private let _batchCapacity: Int
	
	private var _layers: [Layer]
	private var _terminalLayer: TerminalLayer
	private var _optimizer: Optimizer
	
	private var _tempBuffer1: FloatBuffer
	private var _tempBuffer2: FloatBuffer

}
