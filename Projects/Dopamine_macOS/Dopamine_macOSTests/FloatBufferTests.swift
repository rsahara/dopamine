//
//  FloatBufferTests.swift
//  Dopamine_macOS
//
//  Created by 佐原瑠能 on 2017/07/23.
//  Copyright © 2017年 Runo. All rights reserved.
//

import XCTest
import Dopamine_macOS

class FloatBufferTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
	
	func testZeroFill() {
		let a = FloatBuffer(10, 1)
		a.fillZero()
		for index in 0 ..< a.capacity {
			XCTAssertEqualWithAccuracy(a.contents[index], 0.0, accuracy: 0.000001)
		}
		
	}

    func testAddSub() {
		let a = FloatBuffer(10, 1)

		let b = FloatBuffer(1, 1)
		b.contents[0] = 1.0
		
		a.fillZero()
		a.add(b)
		for index in 0 ..< a.capacity {
			XCTAssertEqualWithAccuracy(a.contents[index], 1.0, accuracy: 0.000001)
		}
		
		a.fillZero()
		a.add(b, scaledBy: 2.0)
		for index in 0 ..< a.capacity {
			XCTAssertEqualWithAccuracy(a.contents[index], 2.0, accuracy: 0.000001)
		}

		a.fillZero()
		a.add(3.0)
		for index in 0 ..< a.capacity {
			XCTAssertEqualWithAccuracy(a.contents[index], 3.0, accuracy: 0.000001)
		}

		a.fillZero()
		a.sub(b)
		for index in 0 ..< a.capacity {
			XCTAssertEqualWithAccuracy(a.contents[index], -1.0, accuracy: 0.000001)
		}
	}

}
