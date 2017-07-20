//
//  AffineLayer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/01.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

// Affine layer.
public class AffineLayer: Layer {

	#if DEBUG
	public var DEBUG_LOG = false		// Print debugging logs
	#endif

	public init(inputSize: Int, outputSize: Int, batchCapacity: Int, layerName: String = "") {
		_layerName = layerName
		_inputSize = inputSize
		_outputSize = outputSize
		_batchCapacity = batchCapacity
		
		_weight = FloatBuffer(inputSize, outputSize)
		_weight.fillRandom()
		_weight.mul(sqrtf(2.0 / Float(inputSize))) // TODO: make a parameter for this

		_bias = FloatBuffer(1, outputSize)
		_bias.fillZero()

		_lastInput = FloatBuffer(batchCapacity, inputSize)

		_dWeight = FloatBuffer(like: _weight)
		_dBias = FloatBuffer(like: _bias)
		
		_tempBuffer = FloatBuffer(1, max(_lastInput.capacity, _weight.capacity))
	}
	
	public func forwardPredict(input: FloatBuffer, result: FloatBuffer) {
		assert(input._columns == _inputSize)
		assert(input._rows <= _batchCapacity)
		
		input.matmul(by: _weight, to: result)
		result.add(_bias)
		
		assert(result._columns == _outputSize)
		assert(result._rows == input._rows)
	}
	
	public func forwardTrain(input: FloatBuffer, result: FloatBuffer, hasPreviousLayer: Bool) {
		forwardPredict(input: input, result: result)
		_lastInput.copy(from: input)
	}
	
	public func backwardTrain(dOutput: FloatBuffer, result: FloatBuffer, hasPreviousLayer: Bool) {
		assert(dOutput._columns == _outputSize)
		assert(dOutput._rows == _lastInput._rows)
		
		if hasPreviousLayer {
			_weight.transpose(result: _tempBuffer)
			dOutput.matmul(by: _tempBuffer, to: result)
		}
		
		_lastInput.transpose(result: _tempBuffer)
		_tempBuffer.matmul(by: dOutput, to: _dWeight)

		dOutput.sumFirstAxis(to: _dBias)
	}
	
	public func requiredResultCapacity() -> Int {
		return _outputSize * _batchCapacity
	}

	public func initOptimizer(optimizer: Optimizer) {
		optimizer.initialize(context: &_weightOptContext, rows: _weight.rows, columns: _weight.columns)
		optimizer.initialize(context: &_biasOptContext, rows: _bias.rows, columns: _bias.columns)
	}

	public func optimize(optimizer: Optimizer) {
		optimizer.optimize(input: _weight, gradient: _dWeight, context: &_weightOptContext)
		optimizer.optimize(input: _bias, gradient: _dBias, context: &_biasOptContext)

		#if DEBUG
		if DEBUG_LOG {
			let (weightMin, weightMax) = _weight.minMax()
			let (biasMin, biasMax) = _bias.minMax()
			print("AffineLayer \(_layerName): Wmin: \(weightMin) Wmax: \(weightMax) Bmin: \(biasMin) Bmax: \(biasMax)")
		}
		#endif
	}

	public var weight: FloatBuffer { return _weight }
	public var bias: FloatBuffer {  return _bias }
	public var dWeight: FloatBuffer { return _dWeight }
	public var dBias: FloatBuffer { return _dBias }

	// MARK: - Hidden
	
	private let _layerName: String
	private let _inputSize: Int
	private let _outputSize: Int
	private let _batchCapacity: Int
	private let _lastInput: FloatBuffer
	private let _tempBuffer: FloatBuffer
	private let _weight: FloatBuffer
	private let _bias: FloatBuffer
	private let _dWeight: FloatBuffer
	private let _dBias: FloatBuffer
	private var _weightOptContext: AnyObject?
	private var _biasOptContext: AnyObject?
}
