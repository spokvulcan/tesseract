import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @Bindable var viewModel: TTSViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                settingsContent
            }
            .navigationTitle("Settings")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        #if os(macOS)
        .frame(minWidth: 450, minHeight: 700)
        #endif
    }

    #if os(iOS)
    private let sectionSpacing: CGFloat = 12
    private let labelFont: Font = .caption
    private let textFont: Font = .footnote
    private let horizontalPadding: CGFloat = 16
    #else
    private let sectionSpacing: CGFloat = 16
    private let labelFont: Font = .subheadline
    private let textFont: Font = .subheadline
    private let horizontalPadding: CGFloat = 20
    #endif

    private var settingsContent: some View {
        VStack(alignment: .leading, spacing: 0) {
            // Model Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Model")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack(spacing: 6) {
                    TextField("Model ID", text: $viewModel.modelId)
                        .font(textFont)
                        .textFieldStyle(.plain)
                        .padding(8)
                        .background(Color.gray.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 6))

                    Button(action: {
                        Task {
                            await viewModel.reloadModel()
                        }
                    }) {
                        Text("Load")
                            .font(textFont)
                            .fontWeight(.medium)
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 8)
                            .background(Color.blue)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.isLoading)
                }
                .padding(.top, 4)
            }
            .padding(.bottom, sectionSpacing)

            // Length Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Length")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Max Tokens")
                        .font(textFont)
                    Spacer()
                    Text("\(viewModel.maxTokens)")
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.maxTokens) },
                        set: { viewModel.maxTokens = Int($0) }
                    ),
                    range: 100...2000,
                    step: 100
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.maxTokens) },
                        set: { viewModel.maxTokens = Int($0) }
                    ),
                    in: 100...2000,
                    step: 100
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Temperature Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Temperature")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Temperature")
                        .font(textFont)
                    Spacer()
                    Text(String(format: "%.2f", viewModel.temperature))
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.temperature) },
                        set: { viewModel.temperature = Float($0) }
                    ),
                    range: 0.0...1.0,
                    step: 0.05
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.temperature) },
                        set: { viewModel.temperature = Float($0) }
                    ),
                    in: 0.0...1.0,
                    step: 0.05
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Top P Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Top P")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                HStack {
                    Text("Top P")
                        .font(textFont)
                    Spacer()
                    Text(String(format: "%.2f", viewModel.topP))
                        .font(textFont)
                        .foregroundStyle(.secondary)
                }
                .padding(.top, 4)

                #if os(iOS)
                CompactSlider(
                    value: Binding(
                        get: { Double(viewModel.topP) },
                        set: { viewModel.topP = Float($0) }
                    ),
                    range: 0.0...1.0,
                    step: 0.05
                )
                #else
                Slider(
                    value: Binding(
                        get: { Double(viewModel.topP) },
                        set: { viewModel.topP = Float($0) }
                    ),
                    in: 0.0...1.0,
                    step: 0.05
                )
                .tint(.blue)
                #endif
            }
            .padding(.bottom, sectionSpacing)

            // Text Chunking Section
            VStack(alignment: .leading, spacing: 2) {
                Text("Text Chunking")
                    .font(labelFont)
                    .foregroundStyle(.secondary)

                #if os(iOS)
                CompactToggle(label: "Chunk text", isOn: $viewModel.enableChunking, font: textFont)
                    .padding(.top, 4)
                #else
                Toggle("Chunk text", isOn: $viewModel.enableChunking)
                    .font(textFont)
                    .padding(.top, 4)
                #endif

                if viewModel.enableChunking {
                    #if os(iOS)
                    CompactToggle(label: "Stream audio", isOn: $viewModel.streamingPlayback, font: textFont)
                    #else
                    Toggle("Stream audio", isOn: $viewModel.streamingPlayback)
                        .font(textFont)
                    #endif

                    HStack {
                        Text("Max chunk length")
                            .font(textFont)
                        Spacer()
                        Text("\(viewModel.maxChunkLength)")
                            .font(textFont)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.top, 4)

                    #if os(iOS)
                    CompactSlider(
                        value: Binding(
                            get: { Double(viewModel.maxChunkLength) },
                            set: { viewModel.maxChunkLength = Int($0) }
                        ),
                        range: 100...500,
                        step: 50
                    )
                    #else
                    Slider(
                        value: Binding(
                            get: { Double(viewModel.maxChunkLength) },
                            set: { viewModel.maxChunkLength = Int($0) }
                        ),
                        in: 100...500,
                        step: 50
                    )
                    .tint(.blue)
                    #endif

                    TextField("Split pattern", text: $viewModel.splitPattern)
                        .font(textFont)
                        .textFieldStyle(.plain)
                        .padding(8)
                        .background(Color.gray.opacity(0.15))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }

            // Reset button
            Button(action: {
                viewModel.modelId = "mlx-community/VyvoTTS-EN-Beta-4bit"
                viewModel.maxTokens = 1200
                viewModel.temperature = 0.6
                viewModel.topP = 0.8
                viewModel.enableChunking = true
                viewModel.maxChunkLength = 200
                viewModel.splitPattern = "\n"
                viewModel.streamingPlayback = true
            }) {
                Text("Reset to Defaults")
                    .font(textFont)
                    .fontWeight(.medium)
                    .foregroundStyle(.blue)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.blue.opacity(0.15))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .padding(.top, 16)
            .padding(.bottom, 12)
        }
        .padding(.horizontal, horizontalPadding)
    }
}

#Preview {
    SettingsView(viewModel: TTSViewModel())
}
