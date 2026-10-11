//
//  LaunchModeTests.swift
//  tesseractTests
//
//  The **Launch Mode** (ADR-0095): the mode a process's arguments and
//  environment decide, what each mode may do, and the harness table that
//  names the harnesses. Pure: the decision takes its arguments, environment
//  and harness flags as values, so no launch is needed.
//

import Foundation
import Testing

@testable import Tesseract_Agent

struct LaunchModeTests {

    private static let flags = Harness.table.map(\.flag)

    private static func decide(
        _ arguments: [String] = [], _ environment: [String: String] = [:]
    ) -> LaunchMode {
        LaunchMode.decide(
            arguments: ["/Applications/Tesseract Agent.app/Contents/MacOS/Tesseract Agent"]
                + arguments,
            environment: environment, harnessFlags: flags)
    }

    // MARK: - The decision

    @Test func aPlainLaunchIsTheOwners() {
        #expect(Self.decide() == .owner)
        // Xcode's own launch arguments name no harness.
        #expect(Self.decide(["-NSDocumentRevisionsDebugMode", "YES"]) == .owner)
        // A harness option without its harness is not a harness.
        #expect(Self.decide(["--bench-model-id", "qwen3.5-4b"]) == .owner)
    }

    @Test(arguments: LaunchMode.testRunnerVariables)
    func eachTestRunnerVariableMakesATestHost(variable: String) {
        #expect(Self.decide([], [variable: "x"]) == .testHost)
    }

    @Test func theScratchVariableMakesAScratchLaunch() {
        #expect(Self.decide([], ["TESSERACT_SCRATCH_DATA": "1"]) == .scratch)
        #expect(Self.decide([], ["TESSERACT_SCRATCH_DATA": "0"]) == .owner)
    }

    @Test func aHarnessFlagMakesThatHarness() {
        #expect(Self.decide(["--dflash2-bench"]) == .harness(flag: "--dflash2-bench"))
        #expect(
            Self.decide(["--trace-replay", "--bench-model-id", "x"])
                == .harness(flag: "--trace-replay"))
    }

    /// The environment decides first: a test host never runs a harness or
    /// the scratch launch, and a scratch launch never runs a harness.
    @Test func theEnvironmentDecidesBeforeTheArguments() {
        let testRunner = ["XCTestConfigurationFilePath": "/tmp/x.xctestconfiguration"]
        #expect(Self.decide(["--benchmark"], testRunner) == .testHost)
        #expect(
            Self.decide([], testRunner.merging(["TESSERACT_SCRATCH_DATA": "1"]) { $1 })
                == .testHost)
        #expect(Self.decide(["--benchmark"], ["TESSERACT_SCRATCH_DATA": "1"]) == .scratch)
    }

    /// `scripts/bench.sh` always passes `--benchmark`; the harness it is
    /// asked for comes after it on the command line and still wins.
    @Test func aHarnessBenchShStartsWinsOverBenchmark() {
        for flag in [
            "--paro-parity-bench", "--snapshot-bench", "--prefix-detect-bench",
            "--tokenize-cache-bench", "--agent-cpu-bench", "--prefix-cache-e2e",
            "--turboquant-bench", "--dflash2-bench",
        ] {
            #expect(
                Self.decide(["--benchmark", "--bench-sweep", "quick", flag])
                    == .harness(flag: flag), "\(flag)")
        }
        #expect(Self.decide(["--benchmark"]) == .harness(flag: "--benchmark"))
    }

    // MARK: - What each mode may do

    @Test func eachModeAnswersWhatItMayDo() {
        let harness = LaunchMode.harness(flag: "--dflash2-bench")
        // keeps off data, background work, single instance, migrates, notifies, sounds
        let table: [(LaunchMode, [Bool])] = [
            (.owner, [false, true, true, true, true, true]),
            (.scratch, [true, true, true, false, false, true]),
            (.testHost, [true, false, false, false, false, false]),
            (harness, [false, false, false, false, false, true]),
        ]
        for (mode, answers) in table {
            #expect(
                [
                    mode.keepsOffOwnersData, mode.runsOwnersBackgroundWork,
                    mode.enforcesSingleInstance, mode.migratesOwnersData,
                    mode.postsNotifications, mode.playsSounds,
                ] == answers, "\(mode)")
        }
    }

    /// The owner's first launch is the Welcome window's, every later one the
    /// main window's; a scratch launch opens the main window either way; a
    /// test host and a harness open none.
    @Test func theWindowALaunchOpens() {
        #expect(LaunchMode.owner.launchWindow(onboarded: false) == .welcome)
        #expect(LaunchMode.owner.launchWindow(onboarded: true) == .main)
        #expect(LaunchMode.scratch.launchWindow(onboarded: false) == .main)
        #expect(LaunchMode.testHost.launchWindow(onboarded: true) == nil)
        for harness in Harness.table {
            #expect(LaunchMode.harness(flag: harness.flag).launchWindow(onboarded: true) == nil)
        }
    }

    /// Every harness owns the GPU and its model: none runs the owner's
    /// background work (before, only three of them skipped it).
    @Test func noHarnessRunsTheOwnersBackgroundWork() {
        for flag in Self.flags {
            let mode = Self.decide([flag])
            #expect(mode.harness?.flag == flag)
            #expect(!mode.runsOwnersBackgroundWork, "\(flag)")
            #expect(!mode.migratesOwnersData, "\(flag)")
        }
    }

    /// Shared code (`StorageEnvironment`) decides from the environment alone:
    /// whatever the arguments, the answer is the same.
    @Test func keepingOffTheOwnersDataNeverDependsOnTheArguments() {
        let environments: [[String: String]] = [
            [:], ["TESSERACT_SCRATCH_DATA": "1"], ["XCTestSessionIdentifier": "x"],
        ]
        for environment in environments {
            let fromEnvironment = LaunchMode.decide(
                arguments: [], environment: environment, harnessFlags: []
            ).keepsOffOwnersData
            for flag in Self.flags {
                #expect(Self.decide([flag], environment).keepsOffOwnersData == fromEnvironment)
            }
        }
    }

    @Test func theTestHostKnowsItIsOne() {
        #expect(LaunchMode.current == .testHost)
        #expect(StorageEnvironment.keepsOffOwnersData)
    }

    // MARK: - The harness table

    @Test func harnessFlagsAreUniqueFlags() {
        #expect(Set(Self.flags).count == Self.flags.count)
        #expect(Self.flags.allSatisfy { $0.hasPrefix("--") })
        #expect(Set(Harness.table.map(\.label)).count == Harness.table.count)
    }

    /// `--benchmark` logs its failure and exits 0, as it always has; every
    /// other harness fails with 1.
    @Test func onlyTheBenchmarkExitsZeroOnFailure() {
        for harness in Harness.table {
            #expect(harness.failureExitCode == (harness.flag == "--benchmark" ? 0 : 1))
        }
    }
}
