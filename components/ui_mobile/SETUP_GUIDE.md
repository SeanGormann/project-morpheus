# Apple HealthKit Sleep Data Integration

Complete guide to integrating Apple Watch sleep data into your React Native Expo app.

## Prerequisites

- Mac with Xcode installed (14.0+ recommended)
- iPhone with Apple Watch paired
- Node.js 18+
- Your existing Expo project

## Step 1: Install Dependencies

```bash
cd ui_mobile

# Install react-native-health
npx expo install react-native-health

# Generate native iOS project (required for native modules)
npx expo prebuild
```

## Step 2: Configure Xcode Project

After running `prebuild`, open the iOS project in Xcode:

```bash
open ios/uimobile.xcworkspace
```

### 2a. Enable HealthKit Capability

1. Select your project in the left sidebar
2. Select your target (same name as your app)
3. Go to **Signing & Capabilities** tab
4. Click **+ Capability**
5. Search for and add **HealthKit**
6. Check **Clinical Health Records** if you want (optional)

### 2b. Configure Signing

1. In **Signing & Capabilities**, under **Signing**:
   - Check **Automatically manage signing**
   - Select your **Team** (your Apple ID)
   - If you don't have a team, click **Add Account** and sign in with your Apple ID

> **Note**: You do NOT need a paid Apple Developer account ($99/year) for testing on your own device. A free Apple ID works fine.

### 2c. Add Info.plist Entries

Open `ios/uimobile/Info.plist` and add these entries (or verify they exist):

```xml
<key>NSHealthShareUsageDescription</key>
<string>This app needs access to your sleep data to display your sleep patterns and help you understand your rest quality.</string>

<key>NSHealthUpdateUsageDescription</key>
<string>This app does not write health data.</string>
```

Or add via Xcode:
1. Open Info.plist in Xcode
2. Add row: `Privacy - Health Share Usage Description`
3. Add row: `Privacy - Health Update Usage Description`

## Step 3: Copy the Integration Files

Copy these files into your project:

```
ui_mobile/
├── types/
│   └── health.ts          # TypeScript types
├── services/
│   └── healthKitService.ts # HealthKit wrapper
├── hooks/
│   └── useSleepData.ts    # React hook
├── components/
│   └── SleepChart.tsx     # Updated chart component
└── screens/
    └── HomeScreen.tsx     # Example usage
```

## Step 4: Build and Run

### Option A: Via Xcode (Recommended for first run)

1. Connect your iPhone via USB
2. In Xcode, select your iPhone from the device dropdown (top bar)
3. Click the **Play** button (or Cmd+R)
4. First build takes a few minutes
5. Trust the developer certificate on your iPhone:
   - **Settings → General → VPN & Device Management → [Your Apple ID] → Trust**

### Option B: Via Command Line

```bash
# Build and run on connected device
npx expo run:ios --device

# Or specify device name
npx expo run:ios --device "Sean's iPhone"
```

## Step 5: Grant Permissions

When the app launches for the first time:

1. iOS will show the Health permissions dialog
2. Enable **Sleep** data access
3. Tap **Allow**

You can also manually configure in:
**Settings → Health → Data Access & Devices → [Your App]**

## Troubleshooting

### "Untrusted Developer" error
Go to **Settings → General → VPN & Device Management** and trust your developer certificate.

### No sleep data showing
- Make sure you've worn your Apple Watch to bed at least once
- Check that sleep tracking is enabled: **Watch App → Sleep → Track Sleep with Apple Watch**
- Verify data exists in the Health app under **Browse → Sleep**

### Build fails with signing error
- Make sure you're signed into Xcode with your Apple ID
- Check that your Bundle Identifier is unique (e.g., `com.yourname.uimobile`)

### HealthKit not available
- HealthKit only works on real devices, not simulators
- Make sure you're testing on a physical iPhone

## File Structure Overview

```
types/health.ts
├── SleepStage (enum)
├── HealthKitSleepSample
├── SleepSegment
├── SleepSession
├── SleepSummary
├── SleepChartDataPoint
└── UseSleepDataReturn

services/healthKitService.ts
├── initialize()           # Request permissions
├── getSleepSamples()      # Raw data for date range
├── getLastNightSleep()    # Processed last night
├── getSleepHistory()      # Past N nights
└── sessionToChartData()   # Convert for charting

hooks/useSleepData.ts
├── lastNight              # Last night's session
├── history                # Past sessions
├── chartData              # Ready for SleepChart
├── isLoading / error      # Status
├── requestPermission()    # Manual permission request
└── refreshData()          # Pull to refresh

components/SleepChart.tsx
└── Renders sleep stages over time with gradient fill
```

## API Quick Reference

```tsx
// In any component
import { useSleepData } from './hooks/useSleepData';

function MyComponent() {
  const {
    lastNight,      // SleepSession | null
    chartData,      // SleepChartDataPoint[] | null
    isLoading,      // boolean
    error,          // string | null
    refreshData,    // () => Promise<void>
  } = useSleepData();

  if (lastNight) {
    console.log(`Slept ${lastNight.summary.totalSleepMinutes} minutes`);
    console.log(`Deep sleep: ${lastNight.summary.deepMinutes} min`);
    console.log(`Sleep efficiency: ${lastNight.summary.sleepEfficiency}%`);
  }

  return (
    <SleepChart
      accentColor="#92b7c9"
      sleepData={chartData}
      isLoading={isLoading}
      error={error}
    />
  );
}
```

## Next Steps

- Add weekly/monthly aggregation views
- Implement sleep goal tracking
- Add sleep quality scoring
- Create sleep trend analysis
- Export data to CSV/JSON
