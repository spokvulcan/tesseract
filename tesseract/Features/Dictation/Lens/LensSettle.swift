//
//  LensSettle.swift
//  tesseract
//
//  Which words settle when the final pass lands (PRD #612): the Live
//  Preview showed one text while the owner talked, the full pass over the
//  whole take writes another, and the words that differ settle into place
//  in the Lens. A word-level longest common subsequence over the two takes'
//  bare words; every final word outside it settled.
//

import Foundation

nonisolated enum LensSettle {

    /// The final text's token indices that the preview did not show.
    static func settled(preview: String, final: String) -> Set<Int> {
        let old = TakeText.tokens(preview).map(\.bare)
        let new = TakeText.tokens(final).map(\.bare)
        guard !old.isEmpty else { return [] }
        guard !new.isEmpty else { return [] }
        var lcs = [[Int]](
            repeating: [Int](repeating: 0, count: new.count + 1), count: old.count + 1)
        for i in stride(from: old.count - 1, through: 0, by: -1) {
            for j in stride(from: new.count - 1, through: 0, by: -1) {
                lcs[i][j] =
                    old[i] == new[j] ? lcs[i + 1][j + 1] + 1 : max(lcs[i + 1][j], lcs[i][j + 1])
            }
        }
        var kept = Set<Int>()
        var i = 0
        var j = 0
        while i < old.count, j < new.count {
            if old[i] == new[j] {
                kept.insert(j)
                i += 1
                j += 1
            } else if lcs[i + 1][j] >= lcs[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        return Set(new.indices).subtracting(kept)
    }
}
