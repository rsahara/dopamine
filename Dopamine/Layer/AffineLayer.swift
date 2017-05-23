//
//  AffineLayer.swift
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/01.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

class AffineLayer: SimpleLayer {
	
	// weight: (h, w) bias: (1, w)
	init(inputSize: Int, outputSize: Int, layerName: String = "", debugLog: Bool = false) {
		
		weight = FloatBuffer(inputSize, outputSize)
		weight.fillRandom()
		weight.mul(sqrtf(2.0 / Float(inputSize)))

		bias = FloatBuffer(1, outputSize)
		bias.fillZero()
		
		tempBuffer = FloatBuffer(like: weight)
		dWeight = FloatBuffer(like: weight)
		dBias = FloatBuffer(like: bias)
		
		lastInput = FloatBuffer(1, inputSize)
		self.layerName = layerName
		self.debugLog = debugLog

		super.init()
	}

	// x: (1, n) -> (1, n)
	override func forward(input: FloatBuffer, result: FloatBuffer, forTraining: Bool) {
//		let perfCheck = PerfCheck("SimpleAffineLayer: forward")
		
		if (forTraining) {
			lastInput.copy(input)
		}

		input.matmul(by: weight, to: result)
		result.add(bias)
		
//		perfCheck.print()
	}
	
	// (1, n) -> (1, n)
	override func backward(doutput: FloatBuffer, result: FloatBuffer) {
//		let perfCheck = PerfCheck("SimpleAffineLayer: backward")
		
		if hasPreviousLayer {
			weight.transpose(result: tempBuffer)
			doutput.matmul(by: tempBuffer, to: result)
		}

		lastInput.transpose(result: tempBuffer)
		tempBuffer.matmul(by: doutput, to: dWeight)
		
//		dBias.copy(doutput)
		doutput.sumFirstAxis(to: dBias)
		
//		perfCheck.print()
	}
	
	override func initOptimizer(optimizer: Optimizer) {
		optimizer.initialize(context: &weightOptimizationContext)
		optimizer.initialize(context: &biasOptimizationContext)
	}

	override func optimize(optimizer: Optimizer) {
		optimizer.optimize(input: weight, gradient: dWeight, context: &weightOptimizationContext)
		optimizer.optimize(input: bias, gradient: dBias, context: &biasOptimizationContext)

		if (debugLog) {
			let (weightMin, weightMax) = weight.minMax()
			let (biasMin, biasMax) = bias.minMax()
			print("AffineLayer \(layerName): Wmin: \(weightMin) Wmax: \(weightMax) Bmin: \(biasMin) Bmax: \(biasMax)")
		}
	}

	var weight: FloatBuffer
	var bias: FloatBuffer
	var lastInput: FloatBuffer
	
	var tempBuffer: FloatBuffer
	var dWeight: FloatBuffer
	var dBias: FloatBuffer
	
	var weightOptimizationContext: AnyObject?
	var biasOptimizationContext: AnyObject?
	
	var layerName: String
	var debugLog: Bool
}
