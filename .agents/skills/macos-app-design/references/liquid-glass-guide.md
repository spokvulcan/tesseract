# Liquid Glass Design Guide

macOS Tahoe (macOS 26) and SwiftUI 6.2 introduce **Liquid Glass**, a dynamic material that combines the optical properties of glass with a sense of fluidity. Liquid Glass automatically adapts to surrounding elements, lighting conditions, and accessibility settings.

This guide covers best practices and practical examples for adopting Liquid Glass in macOS apps.

## 1. System Components Adopt It Automatically

The easiest way to use Liquid Glass is simply by relying on standard system frameworks. Apple automatically applies Liquid Glass to components like:
- Toolbars (`.toolbar { ... }`)
- Sidebars (`NavigationSplitView`)
- Tab bars (`TabView`)
- System Sheets and Popovers (`.sheet()`, `.popover()`)
- Menus and Standard Controls

**Best Practice:** Do not apply custom `.background` elements (like custom visual effect views or solid colors) behind toolbars and navigation components. Remove existing custom backgrounds so the system Liquid Glass can shine through natively.

## 2. Using Liquid Glass on Custom Views

If you build custom interface elements (such as a floating input bar, a custom badge, or custom controls), you can explicitly apply the Liquid Glass effect. 

Use the `.glassEffect` modifier to give any custom view the Liquid Glass treatment:

```swift
struct FloatingInputBar: View {
    var body: some View {
        HStack {
            TextField("Message...", text: .constant(""))
            Button(action: {}) {
                Image(systemName: "paperplane.fill")
            }
        }
        .padding()
        // Applies the Liquid Glass material masked to a rounded rectangle shape
        .glassEffect(in: RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}
```

## 3. Creating Edge-to-Edge and Scroll-Under Layouts

Liquid Glass looks best when content (like lists or images) scrolls underneath it, letting colors and shapes gracefully blur through the glass.

### A. Extending Scroll Under System Controls

By default, `ScrollView` provides a `.scrollEdgeEffectStyle` which interacts nicely with toolbars and tab bars. Ensure your content touches the edges of the screen, allowing system bars to float naturally.

### B. Safe Area Insets for Custom Controls

If you are placing a custom Liquid Glass view (like a floating bottom bar) above a scrollable list, use `.safeAreaInset(edge:)` instead of placing the views in a `VStack`. This prevents the floating bar from blocking the bottom elements of your scroll view, while still allowing the background of the `safeAreaInset` to be transparent.

**Correct Layout:**
```swift
ScrollView {
    LazyVStack {
        ForEach(items) { item in
            Text(item.name)
        }
    }
}
.safeAreaInset(edge: .bottom) {
    // This bar will float over the content and blur it as it scrolls under
    FloatingInputBar()
        .padding()
}
```

### C. Background Extension Effect

For a full edge-to-edge experience (e.g., hero images underneath sidebars or inspectors), use the `.backgroundExtensionEffect()` on your images. It mirrors and blurs the adjacent content, making it seem like the background stretches underneath standard Liquid Glass panels.

```swift
Image("HeroImage")
    .resizable()
    .aspectRatio(contentMode: .fill)
    .backgroundExtensionEffect()
```

## 4. Morphing and Dynamic Grouping

Liquid Glass elements can fluidly morph into each other during transitions (like a segmented control or custom badges).

To orchestrate animations where Liquid Glass morphs:
1. Wrap elements in a `GlassEffectContainer`.
2. Assign each element a unique `.glassEffectID(_:in:)`.

```swift
GlassEffectContainer {
    ForEach(badges) { badge in
        BadgeView(badge: badge)
            .glassEffectID(badge.id, in: namespace)
    }
}
```

## 5. Accessibility Considerations

Liquid Glass honors system settings natively:
- **Reduced Transparency:** Liquid Glass becomes an opaque fallback color.
- **Reduced Motion:** Morphing animations and dynamic shifting behave as standard crossfades.

By using `.glassEffect(in:)` instead of manually trying to blur a background, you ensure your app respects user preferences seamlessly. Avoid stacking too many Liquid Glass elements ("glass on glass"), which can degrade accessibility and visual hierarchy.

## 6. Building Auto-Expanding Liquid Glass Input Bars

When building custom chat/input bars that float over content and use Liquid Glass, standard SwiftUI `TextField(axis: .vertical)` does not scroll naturally when height-constrained. Instead, use an `NSViewRepresentable` wrapping an `NSTextView`:

1. **Dynamic Height Calculation**: Instead of layout hacks, use `NSLayoutManager` within the `NSViewRepresentable` to correctly measure the text bounds:
```swift
func recalculateHeight(textView: NSTextView) {
    guard let layoutManager = textView.layoutManager,
          let textContainer = textView.textContainer else { return }
    
    layoutManager.ensureLayout(for: textContainer)
    let usedRect = layoutManager.usedRect(for: textContainer)
    
    DispatchQueue.main.async {
        self.parent.dynamicHeight = usedRect.height
    }
}
```

2. **Wrapping Constraints**: Configure the `NSTextView` properly so text wraps within the view bounds instead of extending horizontally:
```swift
textView.isVerticallyResizable = true
textView.isHorizontallyResizable = false
textView.autoresizingMask = [.width]
textView.textContainer?.containerSize = NSSize(width: scrollView.contentSize.width, height: CGFloat.greatestFiniteMagnitude)
textView.textContainer?.widthTracksTextView = true
```

3. **Smooth Scroll Setup**: Apply `.frame(height: min(max(dynamicHeight, 20), 150))` on the SwiftUI view. This ensures the input bar starts as a single line, expands seamlessly as you type up to ~150 points, and then gracefully initiates the internal scrollbar for any longer content.