//
//  ObjCExceptionsTests.swift
//  tesseractTests
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct ObjCExceptionsTests {

    private struct Boom: Error {}

    @Test func raisedExceptionBecomesThrownError() {
        let raised = #expect(throws: ObjCExceptions.Raised.self) {
            try ObjCExceptions.catching {
                NSException(
                    name: .internalInconsistencyException,
                    reason: "required condition is false: (_auv3 != nil)", userInfo: nil
                ).raise()
            }
        }
        #expect(raised?.name == NSExceptionName.internalInconsistencyException.rawValue)
        #expect(raised?.reason == "required condition is false: (_auv3 != nil)")
    }

    @Test func swiftErrorPassesThrough() {
        #expect(throws: Boom.self) {
            try ObjCExceptions.catching { throw Boom() }
        }
    }

    @Test func valueReturnsWhenNothingRaises() throws {
        let value = try ObjCExceptions.catching { 42 }
        #expect(value == 42)
    }
}
