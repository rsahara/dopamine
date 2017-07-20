//
//  PerfCheck.swift
//  Dopamine
//
//  Created by Runo Sahara on 2017/05/02.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class PerfCheck {
	
	public init(_ name: String = "") {

		_name = name
		_stepArray = Array<Date>()
		_stepNameArray = Array<String>()

		step("begin")
	}
	
	public func step(_ name: String = "") {
		_stepArray.append(Date())
		_stepNameArray.append(name)
	}
	
	public func print() {
		step("print")
		
		let beginTime = _stepArray.first!
		for stepIndex in 1 ..< _stepArray.count {
			let date = _stepArray[stepIndex]
			let stepName = _stepNameArray[stepIndex]
			Swift.print(String(format:"  %06f %@: %@", date.timeIntervalSince(beginTime), _name, stepName))
		}
	}
	
	public func reset() {
		_stepArray = Array<Date>()
		_stepNameArray = Array<String>()
		step("begin")
	}
	
	// MARK: - Hidden
	
	private let _name: String
	private var _stepArray: Array<Date>
	private var _stepNameArray: Array<String>

}
