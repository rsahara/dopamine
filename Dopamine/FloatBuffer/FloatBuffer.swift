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
	
	public init(_ rows: Int, _ columns: Int) {
		
		_rows = rows
		_columns = columns
		_capacity = rows * columns

		_allocationSize = _capacity
		_buffer = Pointer.allocate(capacity: _allocationSize)
		
		if DEBUG_BUFFERINITIALIZATION {
			for index in 0 ..< _capacity {
				_buffer[index] = Float.nan
			}
		}
	}

	public convenience init(like src: FloatBuffer) {
		self.init(src._rows, src._columns)
	}

	deinit {
		_buffer.deallocate(capacity: _allocationSize)
	}

	// MARK: 計算
	
	public func fillZero() {
		_FloatBuffer_FillZero(_buffer, Int32(_capacity))
	}
	
	public func fillRandom() {
		_FloatBuffer_FillRandomGaussian(_buffer, Int32(capacity))
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
		assert(_columns == right._rows)
		res.resetLazy(_rows, right._columns)
		FloatBuffer_MatMul(res._buffer, _buffer, right._buffer, Int32(_rows), Int32(_columns), Int32(right._columns))
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
		result.resetLazy(_columns, _rows)
		FloatBuffer_Transpose(result._buffer, _buffer, Int32(_rows), Int32(_columns))
	}
	
	public func softmax(result: FloatBuffer) {
		result.resetLazy(like: self)
		FloatBuffer_SoftMax(result._buffer, _buffer, Int32(_rows), Int32(_columns))
	}
	
	public func maxPosition() -> Array<Int> {
		
		let length = _columns
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
		
		for vectIndex in stride(from: 0, to: _capacity, by: _columns) {
			for valIndex in vectIndex ..< vectIndex + _columns {
				valuesStr += "\(_buffer[valIndex]),"
			}
			valuesStr += "|"
		}
		Swift.print("Buffer[\(_rows)x\(_columns)] \(valuesStr)")
	}
	
	public func copy(_ src: FloatBuffer) {
		
		resetLazy(like: src)

		memcpy(_buffer, src._buffer, _capacity * 4)
	}

	public func subcopy(_ src: FloatBuffer, startRow: Int, startColumn: Int) {
		assert(startRow + _rows <= src._rows)
		assert(startColumn + _columns <= src._columns)

		for y in 0 ..< _rows {
			memcpy(_buffer + (y * _columns),
			       src._buffer + ((y + startRow) * src._columns + startColumn),
			       _columns * 4)
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
		result.resetLazy(1, _columns)
		FloatBuffer_SumToFirstAxis(result._buffer, _buffer, Int32(_rows), Int32(_columns))

	}
	
	public func copyConcatRows(left: FloatBuffer, right: FloatBuffer) {
		assert(left._rows == right._rows)

		let resColumns = left._columns + right._columns

		resetLazy(left._rows, resColumns)
		assert(_capacity == left._capacity + right._capacity)

		for y in 0 ..< left._rows {
			let headRes = _buffer + (y * resColumns)
			let headLeft = left._buffer + (y * left._columns)
			let headRight = right._buffer + (y * right._columns)
			
			memcpy(headRes, headLeft, left._columns * 4)
			memcpy(headRes + left._columns, headRight, right._columns * 4)
		}
	}

	public func resetLazy(_ rows: Int, _ columns: Int) {
		let capacity = rows * columns
		if (capacity > _allocationSize) {
			_buffer.deallocate(capacity: _allocationSize)
			_allocationSize = capacity
			_buffer = Pointer.allocate(capacity: _allocationSize)
		}

		_rows = rows
		_columns = columns
		_capacity = capacity

		if DEBUG_BUFFERINITIALIZATION {
			for index in 0 ..< _capacity {
				_buffer[index] = Float.nan
			}
		}
	}

	public func resetLazy(like src: FloatBuffer) {
		resetLazy(src._rows, src._columns)
	}

	// MARK: プロパティ

	public var capacity: Int {
		return _capacity
	}
	
	public var contents: Pointer {
		return _buffer
	}

	public var rows: Int {
		return _rows
	}
	
	public var columns: Int {
		return _columns
	}

	// MARK: プライベート

	private var _buffer: Pointer
	private var _rows: Int
	private var _columns: Int
	private var _capacity: Int
	private var _allocationSize: Int
}
