//
//  ObjCExceptionCatcher.h
//  tesseract
//
//  The one Objective-C seam in the app; used through `ObjCExceptions.catching`.
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Runs `block` inside an Objective-C `@try` and returns the `NSException` it
/// raised, or `nil` if it completed normally. Use through
/// `ObjCExceptions.catching` from Swift.
NSException *_Nullable ObjCExceptionCatcherRun(void (NS_NOESCAPE ^block)(void));

NS_ASSUME_NONNULL_END
