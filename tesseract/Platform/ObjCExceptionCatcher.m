//
//  ObjCExceptionCatcher.m
//  tesseract
//

#import "ObjCExceptionCatcher.h"

NSException *ObjCExceptionCatcherRun(void (NS_NOESCAPE ^block)(void)) {
  @try {
    block();
    return nil;
  } @catch (NSException *exception) {
    return exception;
  }
}
