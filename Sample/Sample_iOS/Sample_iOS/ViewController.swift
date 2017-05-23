//
//  ViewController.swift
//  Sample_iOS
//
//  Created by 佐原 瑠能 on 2017/05/23.
//  Copyright © 2017年 Runo. All rights reserved.
//

import UIKit
import Dopamine

class ViewController: UIViewController {

	override func viewDidLoad() {
		super.viewDidLoad()

		testGRU()
	}

	override func didReceiveMemoryWarning() {
		super.didReceiveMemoryWarning()
		// Dispose of any resources that can be recreated.
	}

	// MARK: - GRU test
	
	func testGRU() {
		
		let dataSize = 4
		let cellSize = 512
		let seqCount = 10
		let layersCount = 1
		let gruNet = GruNet(inputSize: dataSize, cellSize: cellSize, outputSize: dataSize, layersCount: layersCount, sequenceLength: seqCount)
		
		var inputArray: [FloatBuffer] = []
		var outputArray: [FloatBuffer] = []
		var resultArray: [FloatBuffer] = []
		
		
		for seqIndex in 0 ..< seqCount {
			let input = FloatBuffer(1, 4)
			input.fillZero()
			input.contents[seqIndex % 4] = 1.0
			inputArray.append(input)
			
			let output = FloatBuffer(1, 4)
			output.fillZero()
			output.contents[(seqIndex + 1) % 4] = 1.0
			outputArray.append(output)
			
			resultArray.append(FloatBuffer(1, 4))
		}
		
		gruNet.predict(inputArray: inputArray, resultArray: resultArray)
		for result in resultArray {
			result.print()
			Swift.print("res: \(result.maxPosition()[0])")
		}
		
		
		let numIterations = 100
		for iterationIndex in 0 ..< numIterations {
			print("== it: \(iterationIndex)")
			gruNet.train(inputArray: inputArray, outputArray: outputArray)
		}
		
		gruNet.predict(inputArray: [inputArray.first!], resultArray: resultArray)
		for result in resultArray {
			result.print()
			Swift.print("res: \(result.maxPosition()[0])")
		}
		
		// test
		//		let state = FloatBuffer(1, 4)
		//		state.fillZero()
		//		let result = FloatBuffer(1, 4)
		//		let softmax = FloatBuffer(1, 4)
		//		for seqIndex in 0 ..< seqCount {
		//			let cell = cellArray[seqIndex]
		//			let input = inputArray[seqIndex]
		//			cell.forward(input: input, result: result, previousState: state, forTraining: true)
		//
		//			result.softmax(result: softmax)
		//
		//			Swift.print("res: \(softmax.maxPosition()[0])")
		//			state.copy(result)
		//		}
		
	}

}

