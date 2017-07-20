//
//  SimpleGruCell.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/10.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

// TODO: generalize
public class GruCell {

	init(inputSize: Int, outputSize: Int) {
		self.inputSize = inputSize
		self.outputSize = outputSize
		
		affineZ = AffineLayer(inputSize: inputSize + outputSize, outputSize: outputSize, batchCapacity: 1, layerName: "affineZ")
		sigmoidZ = SigmoidLayer()
		affineR = AffineLayer(inputSize: inputSize + outputSize, outputSize: outputSize, batchCapacity: 1, layerName: "affineR")
		sigmoidR = SigmoidLayer()
		affineC = AffineLayer(inputSize: inputSize + outputSize, outputSize: outputSize, batchCapacity: 1, layerName: "affineC")
		tanhC = TanhLayer()

		lastOutput = FloatBuffer(1024, outputSize)
		lastPreviousState = FloatBuffer(1024, outputSize)
		last1mZ = FloatBuffer(1024, outputSize)
	}

	func forwardPredict(input: FloatBuffer, result: FloatBuffer, previousState: FloatBuffer) {
		
		let batchSize = input.rows
		
		result.resetLazy(batchSize, outputSize)
		lastPreviousState.copy(previousState)
		
		// X <- Xt | Ht-1
		let X = FloatBuffer(batchSize, inputSize + outputSize)
		X.copyConcatRows(left: input, right: previousState)
		
		// z <- sigmoid(X.Wz + Bz)
		let Z = FloatBuffer(batchSize, outputSize)
		let temp = FloatBuffer(like: Z)
		affineZ.forwardPredict(input: X, result: temp)
		sigmoidZ.forwardPredict(input: temp, result: Z)
		
		// r <- sigmoid(X.Wr + Br)
		let R = FloatBuffer(batchSize, outputSize)
		affineR.forwardPredict(input: X, result: temp)
		sigmoidR.forwardPredict(input: temp, result: R)
		
		// X' <- Xt | r * Ht-1
		let X2 = FloatBuffer(batchSize, inputSize + outputSize)
		R.mul(previousState)
		X2.copyConcatRows(left: input, right: R)
		
		// X'' <- tanh(X'.Wc + Bc)
		affineC.forwardPredict(input: X2, result: temp)
		X2.resetLazy(like: temp)
		tanhC.forwardPredict(input: temp, result: X2)
		
		// Ht <- (1-z) * Ht-1 + z * X''
		result.copy(Z)
		result.mul(-1.0)
		result.add(1.0)
		last1mZ.copy(result)
		result.mul(previousState)
		Z.mul(X2)
		result.add(Z)
	}

	func forwardTrain(input: FloatBuffer, result: FloatBuffer, previousState: FloatBuffer) {
		
		let batchSize = input.rows
		
		result.resetLazy(batchSize, outputSize)
		lastPreviousState.copy(previousState)
		
		// X <- Xt | Ht-1
		let X = FloatBuffer(batchSize, inputSize + outputSize)
		X.copyConcatRows(left: input, right: previousState)
		
		// z <- sigmoid(X.Wz + Bz)
		let Z = FloatBuffer(batchSize, outputSize)
		let temp = FloatBuffer(like: Z)
		affineZ.forwardTrain(input: X, result: temp, hasPreviousLayer: true)
		sigmoidZ.forwardTrain(input: temp, result: Z, hasPreviousLayer: true)
		
		// r <- sigmoid(X.Wr + Br)
		let R = FloatBuffer(batchSize, outputSize)
		affineR.forwardTrain(input: X, result: temp, hasPreviousLayer: true)
		sigmoidR.forwardTrain(input: temp, result: R, hasPreviousLayer: true)
		
		// X' <- Xt | r * Ht-1
		let X2 = FloatBuffer(batchSize, inputSize + outputSize)
		R.mul(previousState)
		X2.copyConcatRows(left: input, right: R)

		// X'' <- tanh(X'.Wc + Bc)
		affineC.forwardTrain(input: X2, result: temp, hasPreviousLayer: true)
		X2.resetLazy(like: temp)
		tanhC.forwardTrain(input: temp, result: X2, hasPreviousLayer: true)

		// Ht <- (1-z) * Ht-1 + z * X''
		result.copy(Z)
		result.mul(-1.0)
		result.add(1.0)
		last1mZ.copy(result)
		result.mul(previousState)
		Z.mul(X2)
		result.add(Z)
	}

	func backwardTrain(doutput: FloatBuffer, result: FloatBuffer, resultState: FloatBuffer) {

		let batchSize = doutput.rows
		resultState.resetLazy(batchSize, outputSize)
		result.resetLazy(batchSize, inputSize)
		
		resultState.copy(doutput)
		resultState.mul(last1mZ)

		let dC = FloatBuffer(like:doutput)
		dC.copy(doutput)
		dC.mul(sigmoidZ.lastOutput)
		let temp1 = FloatBuffer(like: doutput)
		tanhC.backwardTrain(dOutput: dC, result: temp1, hasPreviousLayer: true)
		let temp2 = FloatBuffer(like: doutput)
		affineC.backwardTrain(dOutput: temp1, result: temp2, hasPreviousLayer: true)
		
		temp1.subcopy(temp2, startRow: 0, startColumn: inputSize) // d14 = temp1
		temp1.subcopy(resultState, startRow: 0, startColumn: 0) // d14 = temp1
		temp2.copy(temp1)
		temp2.mul(sigmoidR.lastOutput)  // d18 = temp2
		resultState.add(temp2)

		temp1.mul(lastPreviousState) // d16
		sigmoidR.backwardTrain(dOutput: temp1, result: temp2, hasPreviousLayer: true)
		affineR.backwardTrain(dOutput: temp2, result: temp1, hasPreviousLayer: true) // d20 = temp1

		let temp3 = FloatBuffer(like: doutput)
		temp3.copy(tanhC.lastOutput)
		temp3.sub(lastPreviousState)
		temp3.mul(doutput)	// d11

		sigmoidZ.backwardTrain(dOutput: temp3, result: temp2, hasPreviousLayer: true)
		affineZ.backwardTrain(dOutput: temp2, result: temp3, hasPreviousLayer: true) // d17 = temp3
		
		temp1.add(temp3) // d21 = temp1
		temp2.subcopy(temp1, startRow: 0, startColumn: inputSize) // d23 = temp2
		resultState.add(temp2)
		
		temp2.resetLazy(like: result)
		temp2.subcopy(temp1, startRow: 0, startColumn:0) // d22 = temp2
		result.add(temp2)

		// h~t: tanhC.lastOutput
		// zt: sigmoidZ.lastOutput
		// ht-1: lastPreviousState
		// rt: sigmoidR.lastOutput
	}
 
	let inputSize: Int
	let outputSize: Int
	
	var lastOutput: FloatBuffer
	var lastPreviousState: FloatBuffer
	var last1mZ: FloatBuffer // last (1 - z)

	var affineZ: AffineLayer
	var sigmoidZ: SigmoidLayer
	var affineR: AffineLayer
	var sigmoidR: SigmoidLayer
	var affineC: AffineLayer
	var tanhC: TanhLayer
}
