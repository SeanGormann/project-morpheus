# HomeScreen Documentation

The main interface for configuring sleep sound schedules.

## Core Features

### Timeline Slider
- **Range**: Maps to user's configured sleep schedule (default 10 PM - 8 AM)
- **Normal mode**: ~3 minute increments (step 0.5 on 0-100 scale)
- **Precision mode**: Minute-level control (step 0.167)

### Precision Mode
Activated by holding the slider thumb stationary for 2+ seconds.

**Behavior**:
- Zooms to ±30 minute window around current position
- Shows minute tick marks instead of hour markers
- Time labels update to show zoomed range with full timestamps
- Exits automatically on slider release

**Visual feedback**:
- "PRECISION" badge appears above time display
- Slider container scales up (1.15x height) with accent border
- Gradient glow effect on slider
- Hint text: "Minute-level precision active"

### Scheduled Events
- Add events at selected time via "Add" button
- Events display as chips with time and remove button
- Sorted chronologically
- Locked when schedule is locked

### Sound Check Panel
- Play/pause button for audio test
- Volume slider (0-100%)
- Shows current test sound info

### Lock Mode
- Prevents accidental schedule changes
- Dims controls when active
- Toggle via "Lock-in Schedule" / "Edit Schedule" button

## Props

```typescript
interface HomeScreenProps {
  accentColor: string;      // Theme accent color
  isDark: boolean;          // Dark mode state
  settings: UserSettings;   // App settings (sleep schedule, etc.)
}
```

## Key State

| State | Type | Purpose |
|-------|------|---------|
| `selectedTime` | number | Current slider position (0-100) |
| `isPrecisionMode` | boolean | Precision zoom active |
| `precisionCenter` | number | Center point when precision activates |
| `isLocked` | boolean | Schedule editing locked |
| `events` | ScheduledEvent[] | List of scheduled sound events |

## Time Calculation

```typescript
const calculateTime = (val: number) => {
  const startMinutes = settings.sleepStartHour * 60;
  const minutesToAdd = Math.round((val / 100) * settings.sleepDuration);
  // Returns { formatted, ampm, full }
};
```

## Related Files

- `types.ts` - UserSettings interface
- `App.tsx` - Settings state management
- `SettingsScreen.tsx` - Configure sleep schedule

