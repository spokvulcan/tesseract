import SwiftUI

/// One large telemetry number: 32 pt bold value, optional unit, uppercase
/// kerned label, optional live dot. The Cache page's headline atom —
/// originally the old Dashboard's hero-band unit, kept so any future hero
/// number reads from the same instrument panel.
struct HeroNumber: View {
    let value: String
    let unit: String?
    let label: String
    let isLive: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(.system(size: 32, weight: .bold))
                    .monospacedDigit()
                    .contentTransition(.numericText())
                    .foregroundStyle(.primary)
                if let unit {
                    Text(unit)
                        .font(.system(size: 15, weight: .medium))
                        .foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 5) {
                if isLive {
                    Circle()
                        .fill(.green)
                        .frame(width: 6, height: 6)
                }
                Text(label)
                    .font(.caption2.weight(.semibold))
                    .textCase(.uppercase)
                    .kerning(0.8)
                    .foregroundStyle(isLive ? AnyShapeStyle(.green) : AnyShapeStyle(.secondary))
            }
        }
    }
}
