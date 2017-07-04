//
//  PerfCheck.swift
//  RunoNetTest
//
//  Created by 佐原瑠能 on 2017/05/02.
//  Copyright © 2017 Runo Sahara. All rights reserved.
//

import Foundation

public class PerfCheck {
	
	public init(_ name: String = "") {

		self.name = name
		stepArray = Array<Date>()
		stepNameArray = Array<String>()

		step("begin")
	}
	
	public func step(_ name: String = "") {
		stepArray.append(Date())
		stepNameArray.append(name)
	}
	
	public func print() {
		step("print")
		
		let beginTime = stepArray.first!
		for stepIndex in 1 ..< stepArray.count {
			let date = stepArray[stepIndex]
			let stepName = stepNameArray[stepIndex]
			Swift.print(String(format:"  %06f %@: %@", date.timeIntervalSince(beginTime), name, stepName))
		}
	}
	
	public func reset() {
		stepArray = Array<Date>()
		stepNameArray = Array<String>()
		step("begin")
	}
	
	let name: String
	var stepArray: Array<Date>
	var stepNameArray: Array<String>

}
