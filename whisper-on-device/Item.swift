//
//  Item.swift
//  whisper-on-device
//
//  Created by Bohdan Ivanchenko on 31.01.2026.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
