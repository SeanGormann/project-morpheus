To rebuild locally before building with Xcode on target device:

```bash
cd /Users/seangorman/code-projects/project-morpheus/components/ui_mobile && npx expo prebuild --platform ios --clean
```

Then, open Xcode and rebuild:
- Open ios/SleepSounds.xcworkspace in Xcode
- Select your device as the target
- Build and run (⌘+R)