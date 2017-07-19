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
		
		_weight = FloatBuffer(inputSize, outputSize)
		_weight.fillRandom()
		_weight.mul(sqrtf(2.0 / Float(inputSize)))

		_bias = FloatBuffer(1, outputSize)
		_bias.fillZero()

		_lastInput = FloatBuffer(batchCapacity, inputSize)

		_dWeight = FloatBuffer(like: _weight)
		_dBias = FloatBuffer(like: _bias)
		
		_tempBuffer = FloatBuffer(1, max(_lastInput.capacity, _weight.capacity))

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

		input.matmul(by: _weight, to: result)
		result.add(_bias)

		assert(result._columns == _outputSize)
		assert(result._rows == input._rows)
	}
	
	override func backward(doutput: FloatBuffer, result: FloatBuffer) {
		assert(doutput._columns == _outputSize)
		assert(doutput._rows == _lastInput._rows)

		if hasPreviousLayer {
			_weight.transpose(result: _tempBuffer)
			doutput.matmul(by: _tempBuffer, to: result)
		}

		_lastInput.transpose(result: _tempBuffer)
		_tempBuffer.matmul(by: doutput, to: _dWeight)
		
		doutput.sumFirstAxis(to: _dBias)
	}
	
	override func initOptimizer(optimizer: Optimizer) {
		optimizer.initialize(context: &_weightOptContext, rows: _weight.rows, columns: _weight.columns)
		optimizer.initialize(context: &_biasOptContext, rows: _bias.rows, columns: _bias.columns)
	}

	override func optimize(optimizer: Optimizer) {
		optimizer.optimize(input: _weight, gradient: _dWeight, context: &_weightOptContext)
		optimizer.optimize(input: _bias, gradient: _dBias, context: &_biasOptContext)

		#if DEBUG
		if _debugLog {
			let (weightMin, weightMax) = _weight.minMax()
			let (biasMin, biasMax) = _bias.minMax()
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
	private let _tempBuffer: FloatBuffer
	
	// TODO: private
	let _weight: FloatBuffer
	let _bias: FloatBuffer
	let _dWeight: FloatBuffer
	let _dBias: FloatBuffer
	
	private var _weightOptContext: AnyObject?
	private var _biasOptContext: AnyObject?

	#if DEBUG
	public var _debugLog: Bool
	#endif
}
