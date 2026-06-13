import MLX

nonisolated enum MLXCheckedEvaluation {
    static func withErrors<R>(_ body: (ErrorBox) throws -> R) throws -> R {
        try MLX.withError(body)
    }

    static func eval(_ values: Any...) throws {
        try MLX.checkedEval(values)
    }
}
