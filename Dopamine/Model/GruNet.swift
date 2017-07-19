//
//  GruNet.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/14.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class GruNet {

	public init(inputSize: Int, cellSize: Int, outputSize: Int, layersCount: Int, sequenceLength: Int) {
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.cellSize = cellSize
		self.layersCount = layersCount
		self.sequenceLength = sequenceLength

		cellArray = [GruCell]()
		outputLayerArray = [AffineLayer]()
		softmaxArray = [SoftmaxWithLoss]()
		for sequenceIndex in 0 ..< sequenceLength {
			cellArray.append(GruCell(inputSize: inputSize, outputSize: cellSize))
			for _ in 1 ..< layersCount {
				cellArray.append(GruCell(inputSize: cellSize, outputSize: cellSize))
			}
			softmaxArray.append(SoftmaxWithLoss())
			outputLayerArray.append(AffineLayer(inputSize: cellSize, outputSize: outputSize, batchCapacity: 1, layerName: "o\(sequenceIndex)", debugLog: false))
		}
		
		tempBufferArray1 = []
		tempBufferArray2 = []
		for _ in 0 ..< layersCount {
			tempBufferArray1.append(FloatBuffer(1, 1024 * 1024))
			tempBufferArray2.append(FloatBuffer(1, 1024 * 1024))
		}

		optimizer = OptimizerDescent(learnRate: 0.1)
//		optimizer = OptimizerRmsProp(learnRate: 0.001)
		for layerIndex in 0 ..< layersCount {
			let cell = cellArray[layerIndex]
			
			cell.affineC.initOptimizer(optimizer: optimizer)
			cell.affineR.initOptimizer(optimizer: optimizer)
			cell.affineZ.initOptimizer(optimizer: optimizer)
		}
		
		outputLayerArray[0].initOptimizer(optimizer: optimizer)
	}

	public func predict(inputArray: [FloatBuffer], resultArray: [FloatBuffer]) {
		
		assert(!resultArray.isEmpty)
		assert(!inputArray.isEmpty)

		let batchSize = inputArray[0].rows
		var previousStateArray = tempBufferArray1
		var nextStateArray = tempBufferArray2
		let tempBuffer = FloatBuffer(batchSize, outputSize)

		// Prepare initial state.
		for layerIndex in 0 ..< layersCount {
			let previousState = previousStateArray[layerIndex]
			previousState.resetLazy(batchSize, cellArray[layerIndex].outputSize)
			previousState.fillZero()
		}

		// Process the input.
		for inputIndex in 0 ..< inputArray.count {
			let input = inputArray[inputIndex]
			for layerIndex in 0 ..< layersCount {
				let cell = cellArray[layerIndex]
				
				cell.forward(input: input, result: nextStateArray[layerIndex], previousState: previousStateArray[layerIndex], forTraining: false)
			}

			// Rotate buffers.
			let temp = nextStateArray
			nextStateArray = previousStateArray
			previousStateArray = temp
		}

		// Process the output.
		let outputLayer = outputLayerArray[0]
		let softmax = softmaxArray[0]
		outputLayer.forward(input: previousStateArray[0], result: tempBuffer, forTraining: false)
		softmax.forward(input: tempBuffer, output: resultArray[0], forTraining: false)
		for resultIndex in 1 ..< resultArray.count {
			let input = resultArray[resultIndex - 1]
			let output = resultArray[resultIndex]
			for layerIndex in 0 ..< layersCount {

				let cell = cellArray[layerIndex]

				cell.forward(input: input, result: nextStateArray[layerIndex], previousState: previousStateArray[layerIndex], forTraining: false)
				outputLayer.forward(input: nextStateArray[0], result: tempBuffer, forTraining: false)
				softmax.forward(input: tempBuffer, output: output, forTraining: false)
			}
			
			// Rotate buffers.
			let temp = nextStateArray
			nextStateArray = previousStateArray
			previousStateArray = temp
		}
	}

	public func train(inputArray: [FloatBuffer], outputArray: [FloatBuffer]) {
		
		assert(!outputArray.isEmpty)
		assert(!inputArray.isEmpty)
		assert(inputArray.count == outputArray.count)
		assert(inputArray.count <= sequenceLength) // TODO: fix
		
		let batchSize = inputArray[0].rows
		var previousStateArray = tempBufferArray1
		var nextStateArray = tempBufferArray2
		let tempBuffer = FloatBuffer(batchSize, outputSize)
		let tempBuffer2 = FloatBuffer(batchSize, outputSize)

		// Copy weights.
		for layerIndex in 0 ..< layersCount {
			let srcCell = cellArray[layerIndex]
			for inputIndex in 1 ..< inputArray.count {

				let dstCell = cellArray[inputIndex * layersCount + layerIndex]
				
				dstCell.affineC.weight.copy(srcCell.affineC.weight)
				dstCell.affineC.bias.copy(srcCell.affineC.bias)
				dstCell.affineR.weight.copy(srcCell.affineR.weight)
				dstCell.affineR.bias.copy(srcCell.affineR.bias)
				dstCell.affineZ.weight.copy(srcCell.affineZ.weight)
				dstCell.affineZ.bias.copy(srcCell.affineZ.bias)
			}
		}
		for inputIndex in 1 ..< inputArray.count {
			let dstOutputLayer = outputLayerArray[inputIndex]
			dstOutputLayer.weight.copy(outputLayerArray[0].weight)
			dstOutputLayer.bias.copy(outputLayerArray[0].bias)
		}
		
		// Initial state for forward process.
		for layerIndex in 0 ..< layersCount {
			let previousState = previousStateArray[layerIndex]
			previousState.resetLazy(batchSize, cellArray[layerIndex].outputSize)
			previousState.fillZero()
		}
		
		// Forward.
		for inputIndex in 0 ..< inputArray.count {
			let input = inputArray[inputIndex]
			let output = outputArray[inputIndex]
			let outputLayer = outputLayerArray[inputIndex]

			for layerIndex in 0 ..< layersCount {
				let cell = cellArray[inputIndex * layersCount + layerIndex]
				
				cell.forward(input: input, result: nextStateArray[layerIndex], previousState: previousStateArray[layerIndex], forTraining: true)
			}

			outputLayer.forward(input: nextStateArray[layersCount - 1], result: tempBuffer, forTraining: true)

			let softmax = softmaxArray[inputIndex]
			softmax.forward(input: tempBuffer, output:output, forTraining: true)

			// Rotate buffers.
			let temp = nextStateArray
			nextStateArray = previousStateArray
			previousStateArray = temp
		}
		
		// Initial state for backward process.
		for layerIndex in 0 ..< layersCount {
			let previousState = previousStateArray[layerIndex]
			previousState.resetLazy(inputArray[0].rows, cellArray[layerIndex].outputSize)
			previousState.fillZero()
		}
		
		// Backward
		for outputIndex in stride(from: outputArray.count - 1, to: -1, by: -1) {
			let softmax = softmaxArray[outputIndex]
			let outputLayer = outputLayerArray[outputIndex]
			softmax.backward(result: tempBuffer)

			outputLayer.backward(doutput: tempBuffer, result: tempBuffer2)

			for layerIndex in stride(from: layersCount - 1, to: -1, by: -1) {

				previousStateArray[layerIndex].add(tempBuffer2)

				let cell = cellArray[outputIndex * layersCount + layerIndex]
				cell.backward(doutput: previousStateArray[layerIndex], result: tempBuffer2, resultState: nextStateArray[layerIndex])
			}

			// Rotate buffers.
			let temp = nextStateArray
			nextStateArray = previousStateArray
			previousStateArray = temp
		}

		// Optimize.
		optimizer.updateIteration()

		// Calculate gradients.
		for layerIndex in 0 ..< layersCount {
			let dstCell = cellArray[layerIndex]

			for inputIndex in 1 ..< inputArray.count {
			
				let srcCell = cellArray[inputIndex * layersCount + layerIndex]
				
				dstCell.affineC.dWeight.add(srcCell.affineC.dWeight)
				dstCell.affineC.dBias.add(srcCell.affineC.dBias)
				dstCell.affineR.dWeight.add(srcCell.affineR.dWeight)
				dstCell.affineR.dBias.add(srcCell.affineR.dBias)
				dstCell.affineZ.dWeight.add(srcCell.affineZ.dWeight)
				dstCell.affineZ.dBias.add(srcCell.affineZ.dBias)
			}
		}
		for inputIndex in 1 ..< inputArray.count {
			let srcOutputLayer = outputLayerArray[inputIndex]
			outputLayerArray[0].dWeight.add(srcOutputLayer.dWeight)
			outputLayerArray[0].dBias.add(srcOutputLayer.dBias)
		}
		for layerIndex in 0 ..< layersCount {
			let cell = cellArray[layerIndex]
			
			cell.affineC.optimize(optimizer: optimizer)
			cell.affineR.optimize(optimizer: optimizer)
			cell.affineZ.optimize(optimizer: optimizer)
		}
		outputLayerArray[0].optimize(optimizer: optimizer)

	}

	let inputSize: Int
	let cellSize: Int
	let outputSize: Int
	let layersCount: Int
	let sequenceLength: Int

	var cellArray: [GruCell]
	var outputLayerArray: [AffineLayer]
	var softmaxArray: [SoftmaxWithLoss]
	let optimizer: Optimizer

	var tempBufferArray1: [FloatBuffer]
	var tempBufferArray2: [FloatBuffer]
}
