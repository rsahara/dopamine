//
//  IntBuffer.swift
//  Pods
//
//  Created by 佐原 瑠能 on 2017/05/30.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

class IntBuffer {

	var DEBUG_BUFFERINITIALIZATION = false
	
	public typealias Pointer = UnsafeMutablePointer<Int>

	public init(_ capacity: Int) {
		
		_capacity = capacity
		_allocationSize = _capacity
		_buffer = Pointer.allocate(capacity: _allocationSize)
		
		if DEBUG_BUFFERINITIALIZATION {
			for index in 0 ..< _capacity {
				_buffer[index] = -1
			}
		}
	}
	
	deinit {
		_buffer.deallocate(capacity: _allocationSize)
	}

	// MARK: プロパティ
	
	public var capacity: Int {
		return _capacity
	}
	
	public var contents: Pointer {
		return _buffer
	}

	// MARK: プライベート
	
	private var _buffer: Pointer
	private var _capacity: Int
	private var _allocationSize: Int
}
