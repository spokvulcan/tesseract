//
//  SQLiteDatabase.swift
//  tesseract
//
//  A small typed wrapper over the *system* SQLite (ADR-0035 §8).
//
//  Deliberately not a library. The app has no database dependency and does not
//  gain one here: macOS ships SQLite 3.51 with FTS5, JSON1 and WAL already
//  compiled in (verified 2026-07-12), so `import SQLite3` is the whole cost.
//  This file exists only to make that C API safe to use from Swift — RAII on
//  statements, typed binding, and errors as `throws`.
//
//  Not thread-safe by itself. `MemoryStore` is an `actor` and owns the single
//  connection; nothing else may touch it.
//

import Foundation
import SQLite3

/// SQLite wants to know whether a bound string outlives the call. It does not
/// here (we bind from Swift Strings that live until `step` returns), so every
/// bind is TRANSIENT and SQLite copies.
///
/// `nonisolated(unsafe)` because the app builds with
/// `SWIFT_DEFAULT_ACTOR_ISOLATION = MainActor`, which would otherwise pin this
/// file-scope constant to the main actor. It is an immutable C sentinel value.
private nonisolated(unsafe) let sqliteTransient = unsafeBitCast(
    -1, to: sqlite3_destructor_type.self)

nonisolated struct SQLiteError: Error, CustomStringConvertible {
    let code: Int32
    let message: String
    let sql: String?

    var description: String {
        if let sql {
            return "SQLite error \(code): \(message) — while running: \(sql)"
        }
        return "SQLite error \(code): \(message)"
    }
}

/// One open database. Owned by an actor; never shared.
nonisolated final class SQLiteDatabase {

    private var handle: OpaquePointer?

    init(path: URL) throws {
        var db: OpaquePointer?
        let flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX
        let rc = sqlite3_open_v2(path.path, &db, flags, nil)
        guard rc == SQLITE_OK, let db else {
            let message = db.map { String(cString: sqlite3_errmsg($0)) } ?? "open failed"
            sqlite3_close_v2(db)
            throw SQLiteError(code: rc, message: message, sql: nil)
        }
        self.handle = db

        // WAL: readers never block the writer. The Memory window reads while
        // sleep writes, so this is load-bearing, not hygiene.
        try execute("PRAGMA journal_mode=WAL")
        try execute("PRAGMA synchronous=NORMAL")
        try execute("PRAGMA foreign_keys=ON")
        // A memory store that blocks the UI is worse than one that fails.
        sqlite3_busy_timeout(db, 3_000)
    }

    deinit {
        sqlite3_close_v2(handle)
    }

    // MARK: - Statements

    /// A prepared statement, finalized on scope exit.
    final class Statement {
        fileprivate let stmt: OpaquePointer
        private let db: OpaquePointer
        private let sql: String

        fileprivate init(db: OpaquePointer, sql: String) throws {
            var stmt: OpaquePointer?
            let rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nil)
            guard rc == SQLITE_OK, let stmt else {
                throw SQLiteError(code: rc, message: String(cString: sqlite3_errmsg(db)), sql: sql)
            }
            self.stmt = stmt
            self.db = db
            self.sql = sql
        }

        deinit { sqlite3_finalize(stmt) }

        // — binding (1-indexed, as SQLite wants) —

        @discardableResult
        func bind(_ index: Int32, _ value: String?) -> Statement {
            if let value {
                sqlite3_bind_text(stmt, index, value, -1, sqliteTransient)
            } else {
                sqlite3_bind_null(stmt, index)
            }
            return self
        }

        @discardableResult
        func bind(_ index: Int32, _ value: Int?) -> Statement {
            if let value {
                sqlite3_bind_int64(stmt, index, Int64(value))
            } else {
                sqlite3_bind_null(stmt, index)
            }
            return self
        }

        @discardableResult
        func bind(_ index: Int32, _ value: Double?) -> Statement {
            if let value {
                sqlite3_bind_double(stmt, index, value)
            } else {
                sqlite3_bind_null(stmt, index)
            }
            return self
        }

        @discardableResult
        func bind(_ index: Int32, _ value: Data?) -> Statement {
            if let value {
                _ = value.withUnsafeBytes { raw in
                    sqlite3_bind_blob(
                        stmt, index, raw.baseAddress, Int32(raw.count), sqliteTransient)
                }
            } else {
                sqlite3_bind_null(stmt, index)
            }
            return self
        }

        // — reading (0-indexed, as SQLite wants) —

        func string(_ column: Int32) -> String? {
            guard let c = sqlite3_column_text(stmt, column) else { return nil }
            return String(cString: c)
        }

        func int(_ column: Int32) -> Int {
            Int(sqlite3_column_int64(stmt, column))
        }

        func double(_ column: Int32) -> Double {
            sqlite3_column_double(stmt, column)
        }

        func data(_ column: Int32) -> Data? {
            guard let bytes = sqlite3_column_blob(stmt, column) else { return nil }
            let count = Int(sqlite3_column_bytes(stmt, column))
            guard count > 0 else { return Data() }
            return Data(bytes: bytes, count: count)
        }

        /// Advances one row. Returns false when the result set is exhausted.
        func step() throws -> Bool {
            let rc = sqlite3_step(stmt)
            switch rc {
            case SQLITE_ROW: return true
            case SQLITE_DONE: return false
            default:
                throw SQLiteError(code: rc, message: String(cString: sqlite3_errmsg(db)), sql: sql)
            }
        }

        /// Runs a statement expected to produce no rows.
        func run() throws {
            _ = try step()
        }

        /// Rewinds so the statement can be re-bound and re-run. Reusing one
        /// prepared statement across a loop inside a transaction is the whole
        /// reason batch writes here are cheap.
        func reset() {
            sqlite3_reset(stmt)
            sqlite3_clear_bindings(stmt)
        }
    }

    func prepare(_ sql: String) throws -> Statement {
        guard let handle else {
            throw SQLiteError(code: SQLITE_MISUSE, message: "database is closed", sql: sql)
        }
        return try Statement(db: handle, sql: sql)
    }

    /// Runs one or more statements with no bindings and no results.
    func execute(_ sql: String) throws {
        guard let handle else {
            throw SQLiteError(code: SQLITE_MISUSE, message: "database is closed", sql: sql)
        }
        var error: UnsafeMutablePointer<CChar>?
        let rc = sqlite3_exec(handle, sql, nil, nil, &error)
        guard rc == SQLITE_OK else {
            let message = error.map { String(cString: $0) } ?? "exec failed"
            sqlite3_free(error)
            throw SQLiteError(code: rc, message: message, sql: sql)
        }
    }

    // MARK: - Transactions

    /// All-or-nothing. The write path commits an episode, its embedding, and
    /// its journal line together or not at all — a half-written memory is a
    /// memory that lies.
    func transaction<T>(_ body: () throws -> T) throws -> T {
        try execute("BEGIN IMMEDIATE")
        do {
            let result = try body()
            try execute("COMMIT")
            return result
        } catch {
            try? execute("ROLLBACK")
            throw error
        }
    }

    /// Rows changed by the most recent INSERT/UPDATE/DELETE — the way to tell
    /// whether an `INSERT OR IGNORE` actually inserted.
    var changes: Int {
        guard let handle else { return 0 }
        return Int(sqlite3_changes64(handle))
    }

    var userVersion: Int {
        guard let stmt = try? prepare("PRAGMA user_version"),
            (try? stmt.step()) == true
        else { return 0 }
        return stmt.int(0)
    }

    func setUserVersion(_ version: Int) throws {
        try execute("PRAGMA user_version = \(version)")
    }
}
