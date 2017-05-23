//
//  Optimizer.swift
//  Dopamine_macOS
//
//  Created by 佐原 瑠能 on 2017/05/23.
//  Copyright © 2017年 Runo. All rights reserved.
//

import Foundation

public protocol Optimizer {
	
	func initialize(context: inout AnyObject?)
	func release(context: inout AnyObject?)
	
	func updateIteration()
	func optimize(input: FloatBuffer, gradient: FloatBuffer, context: inout AnyObject?)
	
}
