//
//  AffineLayer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/01.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class AffineLayer: Layer {

	public init(inputSize: Int, outputSize: Int, batchCapacity: Int, layerName: String = "") {
		_layerName = layerName
		_inputSize = inputSize
		_outputSize = outputSize
		_batchCapacity = batchCapacity
		
		weight = FloatBuffer(inputSize, outputSize)
		weight.fillRandom()
		weight.mul(sqrtf(2.0 / Float(inputSize)))

		bias = FloatBuffer(1, outputSize)
		bias.fillZero()
		
		
		_lastInput = FloatBuffer(batchCapacity, inputSize)

		dWeight = FloatBuffer(like: weight)
		dBias = FloatBuffer(like: bias)
		
		tempBuffer = FloatBuffer(1, max(_lastInput.capacity, weight.capacity))

		#if DEBUG
		_debugLog = false
		#endif

		super.init()
	}

	override func forward(input: FloatBuffer, result: FloatBuffer, forTraining: Bool) {
		assert(input._columns == _inputSize)
		assert(input._rows <= _batchCapacity)

		if forTraining {
			_lastInput.copy(input)
		}

		input.matmul(by: weight, to: result)
		result.add(bias)

		assert(result._columns == _outputSize)
		assert(result._rows == input._rows)
	}
	
	override func backward(doutput: FloatBuffer, result: FloatBuffer) {
		assert(doutput._columns == _outputSize)
		assert(doutput._rows == _lastInput._rows)

		if hasPreviousLayer {
			weight.transpose(result: tempBuffer)
			doutput.matmul(by: tempBuffer, to: result)
		}

		_lastInput.transpose(result: tempBuffer)
		tempBuffer.matmul(by: doutput, to: dWeight)
		
		doutput.sumFirstAxis(to: dBias)
	}
	
	override func initOptimizer(optimizer: Optimizer) {
		optimizer.initialize(context: &_weightOptContext, rows: weight.rows, columns: weight.columns)
		optimizer.initialize(context: &_biasOptContext, rows: bias.rows, columns: bias.columns)
	}

	override func optimize(optimizer: Optimizer) {
		optimizer.optimize(input: weight, gradient: dWeight, context: &_weightOptContext)
		optimizer.optimize(input: bias, gradient: dBias, context: &_biasOptContext)

		#if DEBUG
		if _debugLog {
			let (weightMin, weightMax) = weight.minMax()
			let (biasMin, biasMax) = bias.minMax()
			print("AffineLayer \(_layerName): Wmin: \(weightMin) Wmax: \(weightMax) Bmin: \(biasMin) Bmax: \(biasMax)")
		}
		#endif
	}

	// MARK: - Hidden
	
	private let _layerName: String
	private let _inputSize: Int
	private let _outputSize: Int
	private let _batchCapacity: Int
	private let _lastInput: FloatBuffer
	
	// TODO: private
	var weight: FloatBuffer
	var bias: FloatBuffer
	var tempBuffer: FloatBuffer
	var dWeight: FloatBuffer
	var dBias: FloatBuffer
	
	private var _weightOptContext: AnyObject?
	private var _biasOptContext: AnyObject?

	#if DEBUG
	public var _debugLog: Bool
	#endif
}
