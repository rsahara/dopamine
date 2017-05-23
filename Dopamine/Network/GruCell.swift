//
//  SimpleGruCell.swift
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/10.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

public class GruCell {

	init(inputSize: Int, outputSize: Int) {
		self.inputSize = inputSize
		self.outputSize = outputSize
		
		affineZ = AffineLayer(inputSize: inputSize + outputSize, outputSize: outputSize, layerName: "affineZ", debugLog: false)
		sigmoidZ = SigmoidLayer()
		affineR = AffineLayer(inputSize: inputSize + outputSize, outputSize: outputSize, layerName: "affineR", debugLog: false)
		sigmoidR = SigmoidLayer()
		affineC = AffineLayer(inputSize: inputSize + outputSize, outputSize: outputSize, layerName: "affineC", debugLog: false)
		tanhC = TanhLayer()

		lastOutput = FloatBuffer(1024, outputSize)
		lastPreviousState = FloatBuffer(1024, outputSize)
		last1mZ = FloatBuffer(1024, outputSize)
	}

	func forward(input: FloatBuffer, result: FloatBuffer, previousState: FloatBuffer, forTraining: Bool) {
		
		let batchSize = input.shape.first!
		
		result.resetLazy(batchSize, outputSize)
		lastPreviousState.copy(previousState) // forTraining
		
		// X <- Xt | Ht-1
		let X = FloatBuffer(batchSize, inputSize + outputSize)
		X.copyConcatRows(left: input, right: previousState)
		
		// z <- sigmoid(X.Wz + Bz)
		let Z = FloatBuffer(batchSize, outputSize)
		let temp = FloatBuffer(like: Z)
		affineZ.forward(input: X, result: temp, forTraining: forTraining)
		sigmoidZ.forward(input: temp, result: Z, forTraining: forTraining)
		
		// r <- sigmoid(X.Wr + Br)
		let R = FloatBuffer(batchSize, outputSize)
		affineR.forward(input: X, result: temp, forTraining: forTraining)
		sigmoidR.forward(input: temp, result: R, forTraining: forTraining)
		
		// X' <- Xt | r * Ht-1
		let X2 = FloatBuffer(batchSize, inputSize + outputSize)
		R.mul(previousState)
		X2.copyConcatRows(left: input, right: R)

		// X'' <- tanh(X'.Wc + Bc)
		affineC.forward(input: X2, result: temp, forTraining: forTraining)
		X2.resetLazy(like: temp)
		tanhC.forward(input: temp, result: X2, forTraining: forTraining)

		// Ht <- (1-z) * Ht-1 + z * X''
		result.copy(Z)
		result.mul(-1.0)
		result.add(1.0)
		last1mZ.copy(result)
		result.mul(previousState)
		Z.mul(X2)
		result.add(Z)
	}

	func backward(doutput: FloatBuffer, result: FloatBuffer, resultState: FloatBuffer) {

		let batchSize = doutput.shape.first!
		resultState.resetLazy(batchSize, outputSize)
		result.resetLazy(batchSize, inputSize)
		
		resultState.copy(doutput)
		resultState.mul(last1mZ)

		let dC = FloatBuffer(copyOf:doutput)
		dC.mul(sigmoidZ.lastOutput)
		let temp1 = FloatBuffer(like: doutput)
		tanhC.backward(doutput: dC, result: temp1)
		let temp2 = FloatBuffer(like: doutput)
		affineC.backward(doutput: temp1, result: temp2)
		
		temp1.subcopy(temp2, startingPosition: [0, inputSize]) // d14 = temp1
		temp1.subcopy(resultState, startingPosition: [0, 0]) // d14 = temp1
		temp2.copy(temp1)
		temp2.mul(sigmoidR.lastOutput)  // d18 = temp2
		resultState.add(temp2)

		temp1.mul(lastPreviousState) // d16
		sigmoidR.backward(doutput: temp1, result: temp2)
		affineR.backward(doutput: temp2, result: temp1) // d20 = temp1

		let temp3 = FloatBuffer(like: doutput)
		temp3.copy(tanhC.lastOutput)
		temp3.sub(lastPreviousState)
		temp3.mul(doutput)	// d11

		sigmoidZ.backward(doutput: temp3, result: temp2)
		affineZ.backward(doutput: temp2, result: temp3) // d17 = temp3
		
		temp1.add(temp3) // d21 = temp1
		temp2.subcopy(temp1, startingPosition: [0, inputSize]) // d23 = temp2
		resultState.add(temp2)
		
		temp2.resetLazy(like: result)
		temp2.subcopy(temp1, startingPosition: [0, 0]) // d22 = temp2
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
	var last1mZ: FloatBuffer // 最後の1-z

	var affineZ: AffineLayer
	var sigmoidZ: SigmoidLayer
	var affineR: AffineLayer
	var sigmoidR: SigmoidLayer
	var affineC: AffineLayer
	var tanhC: TanhLayer
}
