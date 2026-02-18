# HealthKit Sleep Data Integration

Displays real Apple Watch sleep data in the History screen.

## Files

| File | Purpose |
|------|---------|
| `types/health.ts` | TypeScript types (SleepStage, SleepSession, SleepSummary) |
| `services/healthKitService.ts` | HealthKit wrapper with Promise APIs |
| `hooks/useSleepData.ts` | React hook for data fetching |
| `components/SleepChart.tsx` | Chart component (supports real + mock data) |
| `screens/HistoryScreen.tsx` | Integrated history view |

## Setup

### 1. Install & Prebuild

```bash
npx expo install react-native-health
npx expo prebuild --clean
```

### 2. Configure Xcode

```bash
open ios/SleepSounds.xcworkspace
```

In Xcode:
1. Select target → **Signing & Capabilities**
2. Set **Team** (your Apple ID)
3. Set unique **Bundle Identifier** (e.g., `com.yourname.sleepsounds`)
4. Click **+ Capability** → Add **HealthKit**
5. **DO NOT** check "Clinical Health Records" (requires paid account)

### 3. Build & Run

Connect iPhone via USB, then in Xcode press **Cmd+R** or:

```bash
npx expo run:ios --device
```

First run: trust developer certificate on iPhone at **Settings → General → VPN & Device Management**.

## Usage

```tsx
import { useSleepData } from '../hooks/useSleepData';
import { SleepChart } from '../components/SleepChart';

function MyScreen() {
  const { 
    history, 
    chartData, 
    isLoading, 
    isLoadingMore,
    hasMoreData,
    loadedDays,
    error, 
    refreshData,
    loadMore,
  } = useSleepData();

  // Last night's data
  const lastNight = history[0];
  if (lastNight) {
    console.log(`Sleep: ${lastNight.summary.totalSleepMinutes} min`);
    console.log(`Deep: ${lastNight.summary.deepMinutes} min`);
    console.log(`Efficiency: ${lastNight.summary.sleepEfficiency}%`);
  }

  return (
    <>
      <SleepChart
        accentColor="#92b7c9"
        sleepData={chartData}
        isLoading={isLoading}
        error={error}
      />
      
      {/* Load more history */}
      {hasMoreData && (
        <Button 
          onPress={loadMore} 
          disabled={isLoadingMore}
          title={`Load More (${loadedDays} days loaded)`}
        />
      )}
    </>
  );
}
```

## API Reference

### useSleepData() Hook

| Property | Type | Description |
|----------|------|-------------|
| `lastNight` | `SleepSession \| null` | Most recent sleep session |
| `history` | `SleepSession[]` | Sleep sessions (paginated) |
| `chartData` | `SleepChartDataPoint[] \| null` | Chart-ready data |
| `isLoading` | `boolean` | Initial loading state |
| `isLoadingMore` | `boolean` | Loading more history |
| `loadedDays` | `number` | Days currently loaded (default: 14) |
| `hasMoreData` | `boolean` | More history available |
| `error` | `string \| null` | Error message |
| `hasPermission` | `boolean \| null` | HealthKit permission status |
| `requestPermission()` | `() => Promise<boolean>` | Request HealthKit access |
| `refreshData()` | `() => Promise<void>` | Pull to refresh (resets to 14 days) |
| `loadMore()` | `() => Promise<void>` | Load 14 more days of history |

### SleepSession

```ts
interface SleepSession {
  id: string;
  date: Date;
  bedTime: Date;
  wakeTime: Date;
  totalDurationMinutes: number;
  segments: SleepSegment[];
  summary: SleepSummary;
}

interface SleepSummary {
  totalSleepMinutes: number;
  timeInBedMinutes: number;
  awakeMinutes: number;
  coreMinutes: number;
  deepMinutes: number;
  remMinutes: number;
  sleepEfficiency: number; // percentage
}
```

### SleepChart Props

| Prop | Type | Description |
|------|------|-------------|
| `accentColor` | `string` | Chart accent color |
| `sleepData` | `SleepChartDataPoint[] \| null` | Real HealthKit data |
| `isLoading` | `boolean` | Show loading spinner |
| `error` | `string \| null` | Show error message |
| `isDark` | `boolean` | Dark mode (default: true) |

## Troubleshooting

**No sleep data**: Wear Apple Watch to bed with sleep tracking enabled (Watch app → Sleep → Track Sleep with Apple Watch).

**Permission denied**: Check Settings → Health → Data Access & Devices → [Your App].

**Build fails with signing error**: Ensure "Clinical Health Records" is **unchecked** in HealthKit capability.

**HealthKit not available**: Only works on physical iOS devices, not simulators.

