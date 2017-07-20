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
		sigmoidZ = SigmoidLayer(inputSize: outputSize, batchCapacity: 1)
		affineR = AffineLayer(inputSize: inputSize + outputSize, outputSize: outputSize, batchCapacity: 1, layerName: "affineR")
		sigmoidR = SigmoidLayer(inputSize: outputSize, batchCapacity: 1)
		affineC = AffineLayer(inputSize: inputSize + outputSize, outputSize: outputSize, batchCapacity: 1, layerName: "affineC")
		tanhC = TanhLayer(inputSize: outputSize, batchCapacity: 1)

		lastOutput = FloatBuffer(1024, outputSize)
		lastPreviousState = FloatBuffer(1024, outputSize)
		last1mZ = FloatBuffer(1024, outputSize)
	}

	func forwardPredict(input: FloatBuffer, result: FloatBuffer, previousState: FloatBuffer) {
		
		let batchSize = input.rows
		
		result.reshape(batchSize, outputSize)
		lastPreviousState.copy(from: previousState)
		
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
		X2.reshape(like: temp)
		tanhC.forwardPredict(input: temp, result: X2)
		
		// Ht <- (1-z) * Ht-1 + z * X''
		result.copy(from: Z)
		result.mul(-1.0)
		result.add(1.0)
		last1mZ.copy(from: result)
		result.mul(previousState)
		Z.mul(X2)
		result.add(Z)
	}

	func forwardTrain(input: FloatBuffer, result: FloatBuffer, previousState: FloatBuffer) {
		
		let batchSize = input.rows
		
		result.reshape(batchSize, outputSize)
		lastPreviousState.copy(from: previousState)
		
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
		X2.reshape(like: temp)
		tanhC.forwardTrain(input: temp, result: X2, hasPreviousLayer: true)

		// Ht <- (1-z) * Ht-1 + z * X''
		result.copy(from: Z)
		result.mul(-1.0)
		result.add(1.0)
		last1mZ.copy(from: result)
		result.mul(previousState)
		Z.mul(X2)
		result.add(Z)
	}

	func backwardTrain(doutput: FloatBuffer, result: FloatBuffer, resultState: FloatBuffer) {

		let batchSize = doutput.rows
		resultState.reshape(batchSize, outputSize)
		result.reshape(batchSize, inputSize)
		
		resultState.copy(from: doutput)
		resultState.mul(last1mZ)

		let temp1 = FloatBuffer(1, 1024 * 1024) // TODO: fix size and dont reallocate
		let temp2 = FloatBuffer(1, 1024 * 1024) // TODO: fix size and dont reallocate
		let temp3 = FloatBuffer(1, 1024 * 1024) // TODO: fix size and dont reallocate

		let dC = FloatBuffer(like:doutput)
		dC.copy(from: doutput)
		dC.mul(sigmoidZ.lastOutput)
		tanhC.backwardTrain(dOutput: dC, result: temp1, hasPreviousLayer: true)
		affineC.backwardTrain(dOutput: temp1, result: temp2, hasPreviousLayer: true)
		
		temp1.subcopy(from: temp2, startRow: 0, startColumn: inputSize) // d14 = temp1
		temp1.subcopy(from: resultState, startRow: 0, startColumn: 0) // d14 = temp1
		temp2.copy(from: temp1)
		temp2.mul(sigmoidR.lastOutput)  // d18 = temp2
		resultState.add(temp2)

		temp1.mul(lastPreviousState) // d16
		sigmoidR.backwardTrain(dOutput: temp1, result: temp2, hasPreviousLayer: true)
		affineR.backwardTrain(dOutput: temp2, result: temp1, hasPreviousLayer: true) // d20 = temp1

		temp3.copy(from: tanhC.lastOutput)
		temp3.sub(lastPreviousState)
		temp3.mul(doutput)	// d11

		sigmoidZ.backwardTrain(dOutput: temp3, result: temp2, hasPreviousLayer: true)
		affineZ.backwardTrain(dOutput: temp2, result: temp3, hasPreviousLayer: true) // d17 = temp3
		
		temp1.add(temp3) // d21 = temp1
		temp2.subcopy(from: temp1, startRow: 0, startColumn: inputSize) // d23 = temp2
		resultState.add(temp2)
		
		temp2.reshape(like: result)
		temp2.subcopy(from: temp1, startRow: 0, startColumn:0) // d22 = temp2
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
