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

	public init(referenceOf src: FloatBuffer, startRow: Int, startColumn: Int, rows: Int, columns: Int) {
		assert(startRow < src._rows)
		assert(startColumn < src._columns)
		assert(startRow * src._columns + startColumn + rows * columns <= src._capacity)
		
		_rows = rows;
		_columns = columns
		_capacity = rows * columns
		
		_allocationSize = 0
		_buffer = src._buffer + (startRow * src._columns + startColumn)
	}

	public convenience init(referenceOf src: FloatBuffer, rowIndex: Int) {
		self.init(referenceOf: src, startRow: rowIndex, startColumn: 0, rows: 1, columns: src._columns)
	}

	deinit {
		if (_allocationSize != 0) {
			_buffer.deallocate(capacity: _allocationSize)
		}
	}

	// MARK: 計算
	
	public func fillZero() {
		_FloatBuffer_FillZero(_buffer, Int32(_capacity))
	}
	
	public func fillRandom() {
		_FloatBuffer_FillRandomGaussian(_buffer, Int32(capacity))
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
		// TODO: C++

		for y in 0 ..< _rows {
			memcpy(_buffer + (y * _columns),
			       src._buffer + ((y + startRow) * src._columns + startColumn),
			       _columns * 4)
		}
	}
	
	public func copyConcatRows(left: FloatBuffer, right: FloatBuffer) {
		assert(left._rows == right._rows)
		// TODO: C++

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

	// TODO: 整理
	public func resetLazy(_ rows: Int, _ columns: Int) {
		let capacity = rows * columns
		if (capacity > _allocationSize) {
			if _allocationSize != 0 {
				_buffer.deallocate(capacity: _allocationSize)
			}
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

	internal var _buffer: Pointer
	internal var _rows: Int
	internal var _columns: Int
	internal var _capacity: Int
	internal var _allocationSize: Int
}
