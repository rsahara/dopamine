//
//  FloatBuffer.swift
//  RunoNetTest
//
//  Created by 佐原 瑠能 on 2017/05/02.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

public class FloatBuffer {
	
	var DEBUG_BUFFERINITIALIZATION = false

	public typealias Pointer = UnsafeMutablePointer<Float>
	
	public convenience init(_ shape: Int...) {
		self.init(shape: shape)
	}

	public init(shape: Array<Int>) {
		assert(!shape.isEmpty)
		
		var capacity = 1
		for size in shape {
			capacity *= size
		}
		
		_allocationSize = capacity
		_capacity = capacity
		_shape = shape

		_buffer = Pointer.allocate(capacity: _allocationSize)
		
		if DEBUG_BUFFERINITIALIZATION {
			for index in 0 ..< _capacity {
				_buffer[index] = Float.nan
			}
		}
	}

	public convenience init(copyFrom buffer: Pointer, shape: Int...) {

		self.init(shape: shape)
		
		for index in 0 ..< _capacity {
			_buffer[index] = buffer[index]
		}
	}

	public convenience init(like src: FloatBuffer) {
		self.init(shape: src._shape)
	}

	public convenience init(copyOf src: FloatBuffer) {
		self.init(shape: src._shape)

		for index in 0 ..< _capacity {
			_buffer[index] = src._buffer[index]
		}
	}

	deinit {
		_buffer.deallocate(capacity: _allocationSize)
	}

	subscript(index: Int) -> Float {
		get {
			return _buffer[index]
		}
		set(val) {
			_buffer[index] = val
		}
	}

	// MARK: 計算
	
	public func fillZero() {
		for index in 0 ..< _capacity {
			_buffer[index] = 0.0
		}
	}
	
	public func fillRandom() {
		for index in 0 ..< _capacity {
			let u1 = Float(arc4random()) * (1.0 / Float(UINT32_MAX))
			let u2 = Float(arc4random()) * (1.0 / Float(UINT32_MAX))
			let f1 = sqrtf(-2.0 * logf(u1))
			let f2 = u2 * (2.0 * Float.pi)
			_buffer[index] = f1 * cosf(f2)
		}
	}
	
	public func printDistributionAuto(title: String = "notitle", desiredDivisions: Int = 50) {
		Swift.print("=== \(title) distribution begin")
		let (rangeArray, countArray) = distributionArrayAuto(desiredDivisions: desiredDivisions)
		for index in 0 ..< rangeArray.count {
			Swift.print("\(rangeArray[index]): \(countArray[index])")
			
		}
		Swift.print("=== \(title) distribution end")
	}

	public func distributionArrayAuto(desiredDivisions: Int = 50) -> (Array<Float>, Array<Int>) {
		assert(_capacity > 0)
		
		var minValue: Float = _buffer[0]
		var maxValue: Float = _buffer[0]

		for index in 1 ..< _capacity {
			let val = _buffer[index]
			if (val < minValue) {
				minValue = val
			}
			if (val > maxValue) {
				maxValue = val
			}
		}
		
		var distance: Float = maxValue - minValue
		if (distance == 0.0) {
			distance = 1.0
		}
		let step = distance / Float(desiredDivisions)
		let startIndex = Int((minValue / step).rounded(.down))
		let endIndex = Int((maxValue / step).rounded(.up))

		var counterDict = Dictionary<Int, Int>()
		for index in 0 ..< _capacity {
			let val = _buffer[index]
			let rangeIndex = Int((val / step).rounded(.down))
			let count: Int = counterDict[rangeIndex] != nil ? counterDict[rangeIndex]! + 1 : 1
			counterDict.updateValue(count, forKey: rangeIndex)
		}

		var resIndex = [Float]()
		var resCount = [Int]()
		for rangeIndex in startIndex ..< endIndex {
			resIndex.append(Float(rangeIndex) * step)
			resCount.append(counterDict[rangeIndex] != nil ? counterDict[rangeIndex]! : 0)
		}
		
		return (resIndex, resCount)
	}
	
	public func matmul(by right: FloatBuffer, to res: FloatBuffer) {
		
		// TODO: shape 対応
		assert(_shape.count == 2)
		assert(right._shape.count == 2)
		assert(res._shape.count == 2)
		
		let width = _shape.last!
		let height = _shape.first!
		let rightWidth = right._shape.last!
		let rightHeight = right._shape.first!
		assert(width == rightHeight)
		
		res.resetLazy(height, rightWidth)

		FloatBuffer_MatMul(res._buffer, _buffer, right._buffer, Int32(height), Int32(width), Int32(rightWidth))

	}

	public func add(_ right: FloatBuffer) {
		let rightCapacity = right._capacity
		assert(_capacity % rightCapacity == 0)

		FloatBuffer_Add(_buffer, right._buffer, Int32(_capacity), Int32(right._capacity))
	}

	public func add(_ right: Float) {
		FloatBuffer_ScalarAdd(_buffer, right, Int32(_capacity))
	}
	
	public func sub(_ right: FloatBuffer) {
		let rightCapacity = right._capacity
		assert(_capacity % rightCapacity == 0)

		FloatBuffer_Sub(_buffer, right._buffer, Int32(_capacity), Int32(right._capacity))
	}
	
	public func sub(_ right: Float) {
		FloatBuffer_ScalarSub(_buffer, right, Int32(_capacity))
	}

	public func mul(_ right: FloatBuffer) {
		let rightCapacity = right._capacity
		assert(_capacity % rightCapacity == 0)
		
		FloatBuffer_Mul(_buffer, right._buffer, Int32(_capacity), Int32(right._capacity))
	}
	
	public func mul(_ right: Float) {
		FloatBuffer_ScalarMul(_buffer, right, Int32(_capacity))
	}
	
	public func div(_ right: FloatBuffer) {
		let rightCapacity = right._capacity
		assert(_capacity % rightCapacity == 0)
		
		FloatBuffer_Div(_buffer, right._buffer, Int32(_capacity), Int32(right._capacity))
	}

	public func transpose(result: FloatBuffer) {
		
		// TODO: shapeに対応
		
		assert(_shape.count >= 2)

		let width = _shape.last!
		let height = _shape[_shape.count - 2]

		result.resetLazy(width, height)
		
		FloatBuffer_Transpose(result._buffer, _buffer, Int32(height), Int32(width))
	}
	
	public func softmax(result: FloatBuffer) {
		
		result.resetLazy(like: self)

		assert(_shape.count > 1)
		FloatBuffer_SoftMax(result._buffer, _buffer, Int32(_shape.first!), Int32(_shape.last!))
	}
	
	public func maxPosition() -> Array<Int> {
		
		let length = _shape.last!
		assert(_capacity % length == 0)
		var result = Array<Int>()
		
		var headIndex = 0
		while (headIndex < _capacity) {
			
			var maxIndex = headIndex
			var maxValue = _buffer[headIndex]
			for index in headIndex + 1 ..< headIndex + length {
				let val = _buffer[index]
				if (val > maxValue) {
					maxValue = val
					maxIndex = index
				}
			}
			
			result.append(maxIndex - headIndex)
			
			headIndex += length
		}
		
		return result
	}
	
	public func minMax() -> (Float, Float) {

		var minVal: Float = _buffer[0]
		var maxVal: Float = _buffer[0]
		for index in 1 ..< _capacity {
			let val = _buffer[index]
			if (val > maxVal) {
				maxVal = val
			}
			else if (val < minVal) {
				minVal = val
			}
		}
		
		return (minVal, maxVal)
	}
	
	public func printMinMax(title: String = "notitle") {
		let (min, max) = minMax()
		Swift.print("  \(title) minmax: \(min) / \(max)")
	}

	
	public func crossEntropyError(against right: FloatBuffer) -> Float {
		assert(_capacity == right.capacity)
		
		var sum: Float = 0.0
		
		FloatBuffer_CrossEntropyError(&sum, _buffer, right._buffer, Int32(_capacity))

		return sum
	}
	
	public func print() {
		var valuesStr = ""
		
		let vectSize = _shape.last!
		for vectIndex in stride(from: 0, to: _capacity, by: vectSize) {
			for valIndex in vectIndex ..< vectIndex + vectSize {
				valuesStr += "\(_buffer[valIndex]),"
			}
			valuesStr += "|"
		}
		Swift.print("Buffer[\(_shape)] \(valuesStr)")
	}
	
	public func copy(_ src: FloatBuffer) {
		
		resetLazy(shape: src._shape)

		memcpy(_buffer, src._buffer, _capacity * 4)
	}

	public func subcopy(_ src: FloatBuffer, startingPosition: [Int]) {
		
		assert(src._shape.count == 2) // 未対応
		assert(startingPosition.count == 2) // 未対応
		assert(startingPosition.first! + _shape.first! <= src._shape.first!)
		assert(startingPosition.last! + _shape.last! <= src._shape.last!)

		for y in 0 ..< _shape.first! {
			memcpy(_buffer + (y * _shape.last!),
			       src._buffer + ((y + startingPosition.first!) * src._shape.last! + startingPosition.last!),
			       _shape.last! * 4)
		}
	}

	public func sqrt() {
		FloatBuffer_Sqrt(_buffer, Int32(_capacity))
	}
	
	// TODO: これはレイヤーに特化しすぎ
	public func maskZeroOrNegative(result: FloatBuffer, mask: FloatBuffer) {

		FloatBuffer_ResetZeroOrNegativeAndMakeMask(result._buffer, mask._buffer, _buffer, Int32(_capacity))

	}
	// TODO: これはレイヤーに特化しすぎ
	public func resetZeroOrNegative(result: FloatBuffer) {
		
		FloatBuffer_ResetZeroOrNegative(result._buffer, _buffer, Int32(_capacity))
		
	}
	// TODO: これはレイヤーに特化しすぎ
	public func applyMask(_ mask: FloatBuffer) {
		assert(_capacity == mask._capacity)
		FloatBuffer_ApplyMask(_buffer, mask._buffer, Int32(_capacity));
	}
	
	public func sumFirstAxis(to result: FloatBuffer) {

		// TODO: shapeに対応
		
		assert(_shape.count >= 2)

		let width = _shape.last!
		let height = _shape[_shape.count - 2]
		
		result.resetLazy(1, width)

		FloatBuffer_SumToFirstAxis(result._buffer, _buffer, Int32(height), Int32(width))

	}
	
	public func copyConcatRows(left: FloatBuffer, right: FloatBuffer) {
		
		// TODO: shapeに対応
		assert(left._shape.first! == right._shape.first!)

		let height = left._shape.first!
		let leftWidth = left._shape.last!
		let rightWidth = right._shape.last!
		let resWidth = leftWidth + rightWidth

		resetLazy(height, resWidth)
		assert(_capacity == left._capacity + right._capacity)

		for y in 0 ..< height {
			let headRes = _buffer + (y * resWidth)
			let headLeft = left._buffer + (y * leftWidth)
			let headRight = right._buffer + (y * rightWidth)
			
			memcpy(headRes, headLeft, leftWidth * 4)
			memcpy(headRes + leftWidth, headRight, rightWidth * 4)
		}
	}

	public func resetLazy(shape: Array<Int>) {

		var capacity = 1
		for size in shape {
			capacity *= size
		}
		
		if (capacity > _allocationSize) {
			_buffer.deallocate(capacity: _allocationSize)
			_allocationSize = capacity
			_buffer = Pointer.allocate(capacity: _allocationSize)
		}

		_shape = shape
		_capacity = capacity

		if DEBUG_BUFFERINITIALIZATION {
			for index in 0 ..< _capacity {
				_buffer[index] = Float.nan
			}
		}
	}

	public func resetLazy(_ shape: Int...) {
		resetLazy(shape: shape)
	}
	
	public func resetLazy(like src: FloatBuffer) {
		resetLazy(shape: src._shape)
	}

	// MARK: プロパティ

	var capacity: Int {
		return _capacity
	}
	
	var contents: Pointer {
		return _buffer
	}
	
	var shape: Array<Int> {
		return _shape
	}
	
	// MARK: プライベート

	private var _buffer: Pointer
	private var _shape: Array<Int>
	private var _capacity: Int
	private var _allocationSize: Int
}
