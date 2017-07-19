//
//  Optimizer.swift
//  Dopamine_macOS
//
//  Created by Runo Sahara on 2017/05/23.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public protocol Optimizer {
	
	func initialize(context: inout AnyObject?, rows: Int, columns: Int)
	func release(context: inout AnyObject?)
	
	func updateIteration()
	func optimize(input: FloatBuffer, gradient: FloatBuffer, context: inout AnyObject?)
	
}
