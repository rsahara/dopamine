//
//  FloatBuffer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/02.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class FloatBuffer {
	
	#if DEBUG
	var DEBUG_BUFFERINITIALIZATION = false
	#endif

	public typealias Pointer = UnsafeMutablePointer<Float>
	
	public init(_ rows: Int, _ columns: Int) {
		
		_rows = rows
		_columns = columns
		_capacity = rows * columns

		_allocationSize = _capacity
		_buffer = Pointer.allocate(capacity: _allocationSize)
		
		#if DEBUG
		if DEBUG_BUFFERINITIALIZATION {
			for index in 0 ..< _capacity {
				_buffer[index] = Float.nan
			}
		}
		#endif
	}

	public convenience init(like src: FloatBuffer) {
		self.init(src._rows, src._columns)
	}

	public init(referenceOf pointer: Pointer, rows: Int, columns: Int) {

		_rows = rows;
		_columns = columns
		_capacity = rows * columns

		_allocationSize = 0
		_buffer = pointer
	}

	public convenience init(referenceOf src: FloatBuffer, startRow: Int, startColumn: Int, rows: Int, columns: Int) {
		assert(startRow < src._rows)
		assert(startColumn < src._columns)
		assert(startRow * src._columns + startColumn + rows * columns <= src._capacity)
		
		self.init(referenceOf: src._buffer + (startRow * src._columns + startColumn), rows: rows, columns: columns)
	}

	public convenience init(referenceOf src: FloatBuffer, rowIndex: Int) {
		self.init(referenceOf: src, startRow: rowIndex, startColumn: 0, rows: 1, columns: src._columns)
	}

	deinit {
		if _allocationSize != 0 {
			_buffer.deallocate(capacity: _allocationSize)
		}
	}

	// MARK: - Main features
	
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
		reshape(like: src)
		memcpy(_buffer, src._buffer, _capacity * 4)
	}

	public func subcopy(_ src: FloatBuffer, startRow: Int, startColumn: Int) {
		assert(startRow + _rows <= src._rows)
		assert(startColumn + _columns <= src._columns)
		// TODO: transcode to C++

		for y in 0 ..< _rows {
			memcpy(_buffer + (y * _columns),
			       src._buffer + ((y + startRow) * src._columns + startColumn),
			       _columns * 4)
		}
	}
	
	public func copyConcatRows(left: FloatBuffer, right: FloatBuffer) {
		assert(left._rows == right._rows)
		// TODO: transcode to C++

		let resColumns = left._columns + right._columns

		resetLazy(left._rows, resColumns) // TODO: reshape
		assert(_capacity == left._capacity + right._capacity)

		for y in 0 ..< left._rows {
			let headRes = _buffer + (y * resColumns)
			let headLeft = left._buffer + (y * left._columns)
			let headRight = right._buffer + (y * right._columns)
			
			memcpy(headRes, headLeft, left._columns * 4)
			memcpy(headRes + left._columns, headRight, right._columns * 4)
		}
	}

	public func reshape(_ rows: Int, _ columns: Int) {
		let capacity = rows * columns
		assert(capacity <= _allocationSize)

		_rows = rows
		_columns = columns
		_capacity = capacity

		#if DEBUG
			if DEBUG_BUFFERINITIALIZATION {
				for index in 0 ..< _capacity {
					_buffer[index] = Float.nan
				}
			}
		#endif
	}
	
	public func reshape(like src: FloatBuffer) {
		reshape(src._rows, src._columns)
	}

	// TODO: Remove, use reshape
	public func resetLazy(_ rows: Int, _ columns: Int) {
		let capacity = rows * columns
		if (capacity > _allocationSize) {
			Swift.print("FloatBuffer: resetLazy: realloc")
			if _allocationSize != 0 {
				_buffer.deallocate(capacity: _allocationSize)
			}
			_allocationSize = capacity
			_buffer = Pointer.allocate(capacity: _allocationSize)
		}

		_rows = rows
		_columns = columns
		_capacity = capacity

		#if DEBUG
		if DEBUG_BUFFERINITIALIZATION {
			for index in 0 ..< _capacity {
				_buffer[index] = Float.nan
			}
		}
		#endif
	}

	// TODO: Remove, use reshape
	public func resetLazy(like src: FloatBuffer) {
		resetLazy(src._rows, src._columns)
	}

	// MARK: - Properties

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
	
	// MARK: - Hidden

	internal static let ALIGNMENT = 8
	
	internal var _buffer: Pointer
	internal var _rows: Int
	internal var _columns: Int
//	internal var _stride: Int
	internal var _capacity: Int
	internal var _allocationSize: Int
}


// MARK: - Save/Load

extension FloatBuffer {

	public convenience init(data: Data) {

		let headerBuffer = UnsafeMutableRawPointer.allocate(bytes: FloatBuffer.HEADERSIZE, alignedTo: FloatBuffer.ALIGNMENT)
		defer {
			headerBuffer.deallocate(bytes: FloatBuffer.HEADERSIZE, alignedTo: FloatBuffer.ALIGNMENT)
		}
		
		data.copyBytes(to: headerBuffer.assumingMemoryBound(to: UInt8.self), count: FloatBuffer.HEADERSIZE)
		
		let rows = headerBuffer.load(as: Int32.self)
		let columns = (headerBuffer + 4).load(as: Int32.self)

		self.init(Int(rows), Int(columns))

		data.advanced(by: FloatBuffer.HEADERSIZE).copyBytes(to: UnsafeMutableRawPointer(_buffer).assumingMemoryBound(to: UInt8.self), count: _capacity * 4)
	}

	public convenience init(contentsOf url: URL) throws {
		self.init(data: try Data(contentsOf: url))
	}

	public func write(to path: URL) throws {
		var data = Data(capacity: writeSize())
		write(to: &data)
		try data.write(to: path, options: Data.WritingOptions.atomic)
	}
	
	public func write(to data: inout Data) {

		let headerBuffer = UnsafeMutableRawPointer.allocate(bytes: FloatBuffer.HEADERSIZE, alignedTo: FloatBuffer.ALIGNMENT)
		defer {
			headerBuffer.deallocate(bytes: FloatBuffer.HEADERSIZE, alignedTo: FloatBuffer.ALIGNMENT)
		}
		headerBuffer.storeBytes(of: Int32(_rows), as: Int32.self)
		(headerBuffer + 4).storeBytes(of: Int32(_columns), as: Int32.self)
		
		data.append(UnsafeBufferPointer<UInt8>(start: headerBuffer.assumingMemoryBound(to: UInt8.self), count: FloatBuffer.HEADERSIZE))
		data.append(UnsafeBufferPointer<Float>(start: _buffer, count: _capacity))
	}
	
	public func writeSize() -> Int {
		return FloatBuffer.HEADERSIZE + _capacity
	}
	
	internal static let HEADERSIZE = 64

}

extension Data {
	
	init(floatBuffer: FloatBuffer) {
		
		self.init(capacity: floatBuffer.writeSize())
		
		floatBuffer.write(to: &self)
		
	}
	
}
