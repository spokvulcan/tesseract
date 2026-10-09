//
//  ServerStatusFormatting.swift
//  tesseract
//

import AppKit
import Foundation
import MLXLMCommon
import SwiftUI

func serverEndpointURL(port: Int) -> String {
    "http://127.0.0.1:\(port)"
}

@MainActor
func copyServerEndpointToPasteboard(port: Int) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(serverEndpointURL(port: port), forType: .string)
}

@MainActor
func copyOpenCodeSetupCommandToPasteboard(port: Int) {
    NSPasteboard.general.clearContents()
    NSPasteboard.general.setString(OpenCodeSetupScript.oneLiner(port: port), forType: .string)
}
