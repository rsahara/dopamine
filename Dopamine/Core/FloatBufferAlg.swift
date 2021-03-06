//
//  FloatBufferAlg.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/06/02.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

extension FloatBuffer {
	
	public func dot(_ right: FloatBuffer) -> Float {
		assert(_capacity == right._capacity)
		return _FloatBuffer_DotProduct(_buffer, right._buffer, Int32(_capacity))
	}
	
	public func matmul(by right: FloatBuffer, to res: FloatBuffer) {
		assert(_columns == right._rows)
		res.reshape(_rows, right._columns)
		_FloatBuffer_MatMul(res._buffer, _buffer, right._buffer, Int32(_rows), Int32(_columns), Int32(right._columns))
	}
	
	public func add(_ right: FloatBuffer) {
		assert(_capacity % right._capacity == 0)
		_FloatBuffer_Add(_buffer, right._buffer, Int32(_capacity), Int32(right._capacity))
	}
	
	public func add(_ right: FloatBuffer, scaledBy rightScale: Float) {
		assert(_capacity % right._capacity == 0)
		_FloatBuffer_AddScaled(_buffer, right._buffer, rightScale, Int32(_capacity), Int32(right._capacity))
	}
	
	public func add(_ right: Float) {
		_FloatBuffer_ScalarAdd(_buffer, right, Int32(_capacity))
	}
	
	public func sub(_ right: FloatBuffer) {
		assert(_capacity % right._capacity == 0)
		_FloatBuffer_Sub(_buffer, right._buffer, Int32(_capacity), Int32(right._capacity))
	}
	
	public func mul(_ right: FloatBuffer) {
		assert(_capacity % right._capacity == 0)
		_FloatBuffer_Mul(_buffer, right._buffer, Int32(_capacity), Int32(right._capacity))
	}
	
	public func mul(_ right: Float) {
		_FloatBuffer_ScalarMul(_buffer, right, Int32(_capacity))
	}
	
	public func div(_ right: FloatBuffer) {
		assert(_capacity % right._capacity == 0)
		_FloatBuffer_Div(_buffer, right._buffer, Int32(_capacity), Int32(right._capacity))
	}
	
	public func transpose(result: FloatBuffer) {
		result.reshape(_columns, _rows)
		_FloatBuffer_Transpose(result._buffer, _buffer, Int32(_rows), Int32(_columns))
	}

	public func sqrt() {
		_FloatBuffer_Sqrt(_buffer, Int32(_capacity))
	}
	
	public func norm() -> Float {
		return _FloatBuffer_Norm(_buffer, Int32(_capacity))
	}
	
	@discardableResult public func normalize() -> Float {
		return _FloatBuffer_Normalize(_buffer, Int32(_capacity))
	}
	public func safeNormalize() {
		_FloatBuffer_SafeNormalize(_buffer, Int32(_capacity))
	}

	public func normalizeRows() {
		return _FloatBuffer_NormalizeRows(_buffer, Int32(_rows), Int32(_columns))
	}

	public func dotProductByRows(_ right: FloatBuffer, to res: FloatBuffer) {
		assert(res._capacity >= _rows)
		assert(right._capacity >= _columns)
		_FloatBuffer_DotProductByRows(res._buffer, _buffer, Int32(_rows), Int32(_columns), right._buffer);
	}

	public func sumFirstAxis(to result: FloatBuffer) {
		result.reshape(1, _columns)
		_FloatBuffer_SumToFirstAxis(result._buffer, _buffer, Int32(_rows), Int32(_columns))
	}

	public func softmax(result: FloatBuffer) {
		result.reshape(like: self)
		_FloatBuffer_Softmax(result._buffer, _buffer, Int32(_rows), Int32(_columns))
	}
	
	public func maxPosition() -> Array<Int> {
		// TODO: C++
		
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
		// TODO: C++
		
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
	
	public func crossEntropyError(against right: FloatBuffer) -> Float {
		assert(_capacity == right.capacity)
		return _FloatBuffer_CrossEntropyError(_buffer, right._buffer, Int32(_capacity))
	}

}
