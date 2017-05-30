//
//  ViewController.swift
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/04/27.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Cocoa
import CoreGraphics
import Dopamine

class ViewController: NSViewController {
	
	override func viewDidLoad() {
		super.viewDidLoad()
		
//		testMNIST()
//		testGRU()
		testSkipGram()
	}
	
	override var representedObject: Any? {
		didSet {
			// Update the view, if already loaded.
		}
	}
	
	@IBOutlet weak var imageView: NSImageView!
	
	// MARK: - MNIST test
	
	func testMNIST() {
		
		preloadMNISTImages()
		
		let net = LayerNet(inputSize: 784, hiddenSize: 50, outputSize: 10)
		let numIterations: Int = 10000
		let batchSize: Int = 100
		let epochBatchCount: Int = max(1, trainImagesBuffer.rows / batchSize)
		
		let batchInput = FloatBuffer(batchSize, 784)
		let batchOutput = FloatBuffer(batchSize, 10)
		
		//		loadTrainRandomSamples(maxSamples: batchSize, input: input, output: output)
		var epochPerfCheck = PerfCheck("epoch")
		for iterationIndex in 0 ..< numIterations {
			
			loadTrainRandomSamples(maxSamples: batchSize, input: batchInput, output: batchOutput)
			net.train(input: batchInput, output: batchOutput)
			
			if (iterationIndex % epochBatchCount == epochBatchCount - 1) {
				
				let testSize = testImagesBuffer.rows
				let testInput = testImagesBuffer! //FloatBuffer(testSize, 784)
				let testOutput = testLabelsBuffer!
				let resultBuffer = FloatBuffer(testSize, 10)
				
				loadTrainRandomSamples(maxSamples: testSize, input: testInput, output: testOutput)
				
				net.predict(input: testInput, result: resultBuffer)
				let maxPositionArray = resultBuffer.maxPosition()
				let resCount = resultBuffer.rows
				var correctCount: Int = 0
				for sampleIndex in 0 ..< resCount {
					let correct = testOutput.contents[sampleIndex * 10 + maxPositionArray[sampleIndex]] == 1.0
					if correct {
						correctCount += 1
					}
				}
				let loss = testOutput.crossEntropyError(against: resultBuffer)
				
				print("it: \(iterationIndex), loss: \(loss / Float(resCount)), accuracy: \(Float(correctCount) * 100.0 / Float(resCount))%")
				
				epochPerfCheck.print()
				epochPerfCheck = PerfCheck("epoch")
			}
		}
	}
	
	// 画像一覧取得
	func loadImagePaths() -> (Dictionary<String, Int>, Dictionary<String, Int>) {
		
		let rootPath = "/Users/rsahara/mnist_png"
		
		var trainArray = Dictionary<String, Int>()
		var testArray = Dictionary<String, Int>()
		
		for value in 0 ..< 10 {
			let fileArray = try! FileManager.default.contentsOfDirectory(atPath: "\(rootPath)/training/\(value)")
			for file in fileArray {
				trainArray["\(rootPath)/training/\(value)/\(file)"] = value
			}
		}
		
		for value in 0 ..< 10 {
			let fileArray = try! FileManager.default.contentsOfDirectory(atPath: "\(rootPath)/testing/\(value)")
			for file in fileArray {
				testArray["\(rootPath)/testing/\(value)/\(file)"] = value
			}
		}
		
		return (trainArray, testArray)
	}
	
	// ランダムでロード
	func loadRandomSamples(filePathDict: Dictionary<String, Int>, filePathArray: Array<String>, maxSamples: Int, input: FloatBuffer, output: FloatBuffer) {
		
		var numSamples = maxSamples
		if (filePathArray.count < numSamples) {
			numSamples = filePathArray.count
		}
		
		var remainFilePathArray = Array<String>(filePathArray)
		var choosenFileArray = Array<String>()
		for _ in 0 ..< maxSamples {
			let randIndex = Int(arc4random_uniform(UInt32(remainFilePathArray.count)))
			let filePath = filePathArray[randIndex]
			
			remainFilePathArray[randIndex] = remainFilePathArray.last!
			remainFilePathArray.removeLast()
			choosenFileArray.append(filePath)
		}
		
		loadImageArray(imagePathArray: choosenFileArray, to: input)
		
		var outputArray = Array<Int>()
		for filePath in choosenFileArray {
			outputArray.append(filePathDict[filePath]!)
		}
		
		loadOneHotArray(valueArray: outputArray, size: 10, to: output)
	}
	
	// 画像のロード
	func loadImage(imagePath: String, to output: FloatBuffer) {
		
		let image = NSImage(byReferencingFile: imagePath)!
		
		let width = Int(image.size.width + 0.5)
		let height = Int(image.size.height + 0.5)
		
		output.resetLazy(1, height * width)
		
		let colorSpace = CGColorSpaceCreateDeviceRGB()
		let context = CGContext(data: output.contents, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 4 * width, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
		
		let graphicsContext = NSGraphicsContext(cgContext: context, flipped: false)
		NSGraphicsContext.setCurrent(graphicsContext)
		image.draw(in: NSRect(x: 0, y: 0, width: width, height: height))
		NSGraphicsContext.setCurrent(nil)
		
		let rgbHead = UnsafeRawPointer(output.contents).assumingMemoryBound(to: UInt8.self)
		for index in 0 ..< output.capacity {
			output.contents[index] = Float(rgbHead[4 * index]) * Float(1.0 / 255.0)
		}
		
		#if DEBUG
			//		for y in 0 ..< height {
			//			var row = ""
			//
			//			for x in 0 ..< width {
			//
			//				let v = output.contents[y * width + x]
			//				let n = Int(v * 255.0)
			//				row.append(String(format: "%02x", n))
			//			}
			//
			//			print(row)
			//		}
			//		print("")
		#endif
		
	}
	
	// 画像のロード
	func loadImageArray(imagePathArray: [String], to output: FloatBuffer) {
		
		let colorSpace = CGColorSpaceCreateDeviceRGB()
		
		let firstImage = NSImage(byReferencingFile: imagePathArray[0])!
		let width = Int(firstImage.size.width + 0.5)
		let height = Int(firstImage.size.height + 0.5)
		let outputUnitSize = width * height
		
		output.resetLazy(imagePathArray.count, height * width)
		
		var outputOffset: Int = 0
		for filePath in imagePathArray {
			
			let image = NSImage(byReferencingFile: filePath)!
			let outputBuffer = output.contents + outputOffset
			
			let context = CGContext(data: outputBuffer, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 4 * width, space: colorSpace, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
			let graphicsContext = NSGraphicsContext(cgContext: context, flipped: false)
			NSGraphicsContext.setCurrent(graphicsContext)
			image.draw(in: NSRect(x: 0, y: 0, width: width, height: height))
			
			let rgbHead = UnsafeRawPointer(outputBuffer).assumingMemoryBound(to: UInt8.self)
			for index in 0 ..< outputUnitSize {
				outputBuffer[index] = Float(rgbHead[4 * index]) * Float(1.0 / 255.0)
			}
			
			outputOffset += outputUnitSize
		}
		
		NSGraphicsContext.setCurrent(nil)
		#if DEBUG
			//		for y in 0 ..< height {
			//			var row = ""
			//
			//			for x in 0 ..< width {
			//
			//				let v = output.contents[y * width + x]
			//				let n = Int(v * 255.0)
			//				row.append(String(format: "%02x", n))
			//			}
			//
			//			print(row)
			//		}
			//		print("")
		#endif
	}
	
	func loadOneHot(value: Int, size: Int, to buffer: FloatBuffer) {
		assert(value < size)
		assert(size <= buffer.capacity)
		
		buffer.fillZero()
		buffer.contents[value] = 1.0
	}
	
	func loadOneHotArray(valueArray: [Int], size: Int, to buffer: FloatBuffer) {
		assert(valueArray.count * size <= buffer.capacity)
		
		buffer.fillZero()
		for valueIndex in 0 ..< valueArray.count {
			let value = valueArray[valueIndex]
			assert(value < size)
			buffer.contents[size * valueIndex + value] = 1.0
		}
	}
	
	func preloadMNISTImages() {
		
		let rootPath = "/Users/rsahara/mnist_png"
		
		trainImagesBuffer = FloatBuffer(60000, 784)
		trainLabelsBuffer =  FloatBuffer(60000, 10)
		trainLabelsBuffer.fillZero()
		
		let trainImagesFileData = try! Data(contentsOf: URL(fileURLWithPath: rootPath + "/train-images-idx3-ubyte"))
		let trainLabelsFileData = try! Data(contentsOf: URL(fileURLWithPath: rootPath + "/train-labels-idx1-ubyte"))
		
		trainImagesFileData.withUnsafeBytes { (pointer: UnsafePointer<UInt8>) in
			let fileHead = pointer + 16
			let bufferHead = trainImagesBuffer.contents
			for pointIndex in 0 ..< trainImagesBuffer.capacity {
				bufferHead[pointIndex] = Float(fileHead[pointIndex]) * Float(1.0 / 255.0)
			}
		}
		
		trainLabelsFileData.withUnsafeBytes { (pointer: UnsafePointer<UInt8>) in
			let fileHead = pointer + 8
			let bufferHead = trainLabelsBuffer.contents
			for imageIndex in 0 ..< trainLabelsBuffer.rows {
				let label = Int(fileHead[imageIndex])
				bufferHead[imageIndex * 10 + label] = 1.0
			}
		}
		
		testImagesBuffer = FloatBuffer(10000, 784)
		testLabelsBuffer =  FloatBuffer(10000, 10)
		testLabelsBuffer.fillZero()
		
		let testImagesFileData = try! Data(contentsOf: URL(fileURLWithPath: rootPath + "/t10k-images-idx3-ubyte"))
		let testLabelsFileData = try! Data(contentsOf: URL(fileURLWithPath: rootPath + "/t10k-labels-idx1-ubyte"))
		
		testImagesFileData.withUnsafeBytes { (pointer: UnsafePointer<UInt8>) in
			let fileHead = pointer + 16
			let bufferHead = testImagesBuffer.contents
			for pointIndex in 0 ..< testImagesBuffer.capacity {
				bufferHead[pointIndex] = Float(fileHead[pointIndex]) * Float(1.0 / 255.0)
			}
		}
		
		testLabelsFileData.withUnsafeBytes { (pointer: UnsafePointer<UInt8>) in
			let fileHead = pointer + 8
			let bufferHead = testLabelsBuffer.contents
			for imageIndex in 0 ..< testLabelsBuffer.rows {
				let label = Int(fileHead[imageIndex])
				bufferHead[imageIndex * 10 + label] = 1.0
			}
		}
		
		
		//		for y in 0 ..< 28 {
		//			var row = ""
		//
		//			for x in 0 ..< 28 {
		//				let n = Int(trainImagesBuffer.contents[y * 28 + x] * 255.0)
		//				row.append(String(format: "%02x", n))
		//			}
		//
		//			print(row)
		//		}
		//		print("")
	}
	
	func loadTrainRandomSamples(maxSamples: Int, input: FloatBuffer, output: FloatBuffer) {
		
		let totalSamples = trainImagesBuffer.rows
		var numSamples = maxSamples
		if (numSamples > totalSamples) {
			numSamples = totalSamples
		}
		
		let imageSize = trainImagesBuffer.columns
		let labelSize = trainLabelsBuffer.columns
		
		var remainingIndexArray = [Int]()
		for imageIndex in 0 ..< totalSamples {
			remainingIndexArray.append(imageIndex)
		}
		for sampleIndex in 0 ..< numSamples {
			let randIndex = Int(arc4random_uniform(UInt32(remainingIndexArray.count)))
			let randImageIndex = remainingIndexArray[randIndex]
			
			memcpy(input.contents + (imageSize * sampleIndex), trainImagesBuffer.contents + (imageSize * randImageIndex), imageSize * 4)
			memcpy(output.contents + (labelSize * sampleIndex), trainLabelsBuffer.contents + (labelSize * randImageIndex), labelSize * 4)
			
			remainingIndexArray[randIndex] = remainingIndexArray.last!
			remainingIndexArray.removeLast()
		}
		
		//		for y in 0 ..< 28 {
		//			var row = ""
		//
		//			for x in 0 ..< 28 {
		//				let n = Int(input.contents[y * 28 + x] * 255.0)
		//				row.append(String(format: "%02x", n))
		//			}
		//
		//			print(row)
		//		}
		//		print("")
	}
	
	var trainImagesBuffer: FloatBuffer!
	var trainLabelsBuffer: FloatBuffer!
	var testImagesBuffer: FloatBuffer!
	var testLabelsBuffer: FloatBuffer!
	
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
	
	// MARK: - SkipGram test
	
	struct CategoryModel {
		public var id: Int
		public var text: String
		public var parentId: Int
		
	}

	func testSkipGram() {
		
		// データをロード
		let categoryModelDict: Dictionary<Int, CategoryModel> = loadCategoryModels()
		
		var groupArrayDict = Dictionary<Int, Array<Int>>()
		for (_, categoryModel) in categoryModelDict {
			if categoryModel.parentId != -1 {
				
				if var array = groupArrayDict[categoryModel.parentId] {
					array.append(categoryModel.id)
					groupArrayDict[categoryModel.parentId] = array
				} else {
					groupArrayDict[categoryModel.parentId] = [categoryModel.parentId, categoryModel.id]
				}

			}
		}

		Swift.print(groupArrayDict)

		var itemSequenceArray = [[Int]]()
		for (_, categoryModel) in categoryModelDict {
			if categoryModel.parentId != -1 {
				itemSequenceArray.append([categoryModel.id, categoryModel.parentId])
			}
		}
		
		let itemVectorSize = 100

		let skipGram = SkipGram(itemCapacity: categoryModelDict.count, itemVectorSize: itemVectorSize)
//		skipGram.trainWithSequences(itemSequenceArray: itemSequenceArray)
		skipGram.trainWithSequences(itemSequenceArray: [[Int]](groupArrayDict.values))
		
		let vectors: FloatBuffer = skipGram.weight

		// Normalize
		for vectorIndex in 0 ..< categoryModelDict.count {
			let vectorHead = vectors.contents + (itemVectorSize * vectorIndex)
			var normsq: Float = 0.0
			for featureIndex in 0 ..< itemVectorSize {
				normsq += vectorHead[featureIndex] * vectorHead[featureIndex]
			}
			let normInv: Float = 1.0 / sqrtf(normsq)
			for featureIndex in 0 ..< itemVectorSize {
				vectorHead[featureIndex] *= normInv
			}
		}

		let testItemIndex = 225//52
		let testItemHead = vectors.contents + (itemVectorSize * testItemIndex)
		var testSimilarityArray = [(Int, Float, String)]()
		for vectorIndex in 0 ..< categoryModelDict.count {
			if let categoryModel = categoryModelDict[vectorIndex] {
				let vectorHead = vectors.contents + (itemVectorSize * vectorIndex)
				
				var dot: Float = 0.0
				for featureIndex in 0 ..< itemVectorSize {
					dot += testItemHead[featureIndex] * vectorHead[featureIndex]
				}

				testSimilarityArray.append((vectorIndex, dot, categoryModel.text))
			}
		}
		
		testSimilarityArray.sort { (a, b) -> Bool in
			return a.1 < b.1
		}
		
		for (index, similarity, text) in testSimilarityArray {
			Swift.print("[\(index)] \(text): \(similarity)")
		}
	}
	
	func loadCategoryModels() -> Dictionary<Int, CategoryModel> {
		let sampleMasterUrl = Bundle.main.url(forResource: "samplemaster", withExtension: "json")!
		let sampleMasterData = try! Data(contentsOf: sampleMasterUrl)
		let sampleMasterObj = (try! JSONSerialization.jsonObject(with: sampleMasterData, options: JSONSerialization.ReadingOptions())) as! Dictionary<String, Any>
		
		let categoryObjArray = sampleMasterObj["genre_m"] as! Array<Dictionary<String, Any>>
		var categoryModelDict = Dictionary<Int, CategoryModel>()
		var translateIdDict = Dictionary<Int, Int>()
		translateIdDict[0] = -1
		var currentId = 0

		for categoryObj in categoryObjArray {
			var categoryModel = CategoryModel(id: Int(categoryObj["genre_id"] as! String)!,
			                                  text: categoryObj["genre_name"] as! String,
			                                  parentId: Int(categoryObj["parent_genre_id"] as! String)!)

			if let newId = translateIdDict[categoryModel.id] {
				categoryModel.id = newId
			} else {
				translateIdDict[categoryModel.id] = currentId
				categoryModel.id = currentId
				currentId += 1
			}

			if let newId = translateIdDict[categoryModel.parentId] {
				categoryModel.parentId = newId
			} else {
				translateIdDict[categoryModel.parentId] = currentId
				categoryModel.parentId = currentId
				currentId += 1
			}

			categoryModelDict.updateValue(categoryModel, forKey: categoryModel.id)
		}

		return categoryModelDict
	}
	
	

}
