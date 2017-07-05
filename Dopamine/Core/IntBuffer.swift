//
//  IntBuffer.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/30.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class IntBuffer {

	#if DEBUG
	var DEBUG_BUFFERINITIALIZATION = false
	#endif
	
	public typealias Pointer = UnsafeMutablePointer<Int32>

	public init(_ capacity: Int) {

		_capacity = capacity
		_allocationSize = _capacity
		_buffer = Pointer.allocate(capacity: _allocationSize)

		#if DEBUG
		if DEBUG_BUFFERINITIALIZATION {
			for index in 0 ..< _capacity {
				_buffer[index] = -1
			}
		}
		#endif
	}
	
	public init(referenceOf pointer: Pointer, capacity: Int) {
		
		_capacity = capacity		
		_allocationSize = 0
		_buffer = pointer
	}

	public convenience init(referenceOf src: IntBuffer, startOffset: Int, capacity: Int) {
		assert(startOffset + capacity <= src._capacity)
		
		self.init(referenceOf: src._buffer + startOffset, capacity: capacity)
	}

	deinit {
		if _allocationSize != 0 {
			_buffer.deallocate(capacity: _allocationSize)
		}
	}
	
	public func print() {
		var valuesStr = ""
		
		for valIndex in 0 ..< _capacity {
			valuesStr += "\(_buffer[valIndex]),"
		}
		Swift.print("Buffer[\(_capacity)] \(valuesStr)")
	}
	
	// MARK: Properties
	
	public var capacity: Int {
		return _capacity
	}
	
	public var contents: Pointer {
		return _buffer
	}

	// MARK: Private

	private var _buffer: Pointer
	private var _capacity: Int
	private var _allocationSize: Int
}
