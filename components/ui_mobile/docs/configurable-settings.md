# Configurable Settings Documentation

This document describes the configurable settings system implemented in the SleepSounds mobile app. All time-related settings that were previously hardcoded have been made user-configurable through the Settings screen.

## Overview

The app now supports fully configurable sleep schedules and audio settings, allowing users to customize their experience based on their personal sleep patterns and preferences.

## Settings Structure

All settings are managed through the `UserSettings` interface defined in `types.ts`:

```typescript
interface UserSettings {
  darkMode: boolean;
  fadeOutTimer: number;        // minutes (max 10)
  fadeInTimer: number;          // minutes (max 10)
  signalDuration: number;       // minutes
  secondaryColor: string;
  sleepStartHour: number;        // 24-hour format (0-23)
  sleepDuration: number;         // minutes
}
```

## Sleep Schedule Settings

### Sleep Start Hour

**Location**: Settings → Sleep Schedule → Sleep Start Time

**Description**: Configures the starting time for the sleep schedule timeline.

**Default**: 22 (10 PM)

**Range**: 0-23 (24-hour format)

**Implementation**:
- Users can select any hour of the day using a modal picker
- The timeline slider in HomeScreen dynamically adjusts to show the selected start time
- Time labels on the slider update automatically based on the configured schedule
- The SleepChart component generates data points based on the configured schedule

**Files Modified**:
- `types.ts`: Added `sleepStartHour` to `UserSettings`
- `App.tsx`: Added default value
- `screens/SettingsScreen.tsx`: Added UI and modal for editing
- `screens/HomeScreen.tsx`: Uses setting for timeline calculation
- `components/SleepChart.tsx`: Uses setting for data generation

### Sleep Duration

**Location**: Settings → Sleep Schedule → Sleep Duration

**Description**: Configures the total duration of the sleep schedule window.

**Default**: 600 minutes (10 hours)

**Range**: 60-1440 minutes (1-24 hours), adjustable in 30-minute increments

**Implementation**:
- Users can adjust duration using a modal picker with 30-minute increments
- The timeline slider spans the configured duration
- All time calculations throughout the app use this duration

**Files Modified**:
- `types.ts`: Added `sleepDuration` to `UserSettings`
- `App.tsx`: Added default value
- `screens/SettingsScreen.tsx`: Added UI and modal for editing
- `screens/HomeScreen.tsx`: Uses setting for timeline calculation
- `components/SleepChart.tsx`: Uses setting for data generation

## Audio Settings

### Fade Out Timer

**Location**: Settings → Appearance → Fade Out Timer

**Description**: Sets how long the audio takes to fade out when stopping.

**Default**: 5 minutes

**Range**: 0-10 minutes

**Implementation**:
- Editable through a modal picker
- Maximum value enforced at 10 minutes
- Can be set to 0 for instant stop

**Files Modified**:
- `screens/SettingsScreen.tsx`: Made editable with modal picker

### Fade In Timer

**Location**: Settings → Appearance → Fade In Timer

**Description**: Sets how long the audio takes to fade in when starting.

**Default**: 5 minutes

**Range**: 0-10 minutes

**Implementation**:
- Editable through a modal picker
- Maximum value enforced at 10 minutes
- Can be set to 0 for instant start

**Files Modified**:
- `screens/SettingsScreen.tsx`: Made editable with modal picker

### Signal Duration

**Location**: Settings → Appearance → Signal Duration

**Description**: Sets how long the sound signal plays when triggered.

**Default**: 30 minutes

**Range**: 1+ minutes (no maximum)

**Implementation**:
- New setting added to the Appearance section
- Editable through a modal picker
- Minimum value of 1 minute enforced
- No maximum limit

**Files Modified**:
- `types.ts`: Added `signalDuration` to `UserSettings`
- `App.tsx`: Added default value
- `screens/SettingsScreen.tsx`: Added UI and modal for editing

## User Interface

### Settings Screen Modals

All time-based settings use a consistent modal interface:

1. **Modal Trigger**: Tap on any editable setting row
2. **Picker Interface**: 
   - Up/Down chevron buttons for increment/decrement
   - Large display of current value
   - Subtitle text explaining the setting
3. **Actions**:
   - Cancel button: Discards changes
   - Save button: Applies changes and closes modal

### Dynamic Updates

All components that use these settings automatically update when settings change:

- **HomeScreen**: Timeline slider, time labels, and calculations update immediately
- **SleepChart**: Chart data points and time labels regenerate based on new schedule
- **HistoryScreen**: Passes settings to SleepChart for consistent display

## Technical Implementation

### State Management

Settings are managed at the App level using React's `useState`:

```typescript
const [settings, setSettings] = useState<UserSettings>({
  // default values
});

const updateSettings = (newSettings: Partial<UserSettings>) => {
  setSettings(prev => ({ ...prev, ...newSettings }));
};
```

Settings are passed down to child components as props, ensuring a single source of truth.

### Data Flow

1. User opens Settings screen
2. User taps on a setting row
3. Modal opens with current value
4. User adjusts value using picker
5. User taps Save
6. `updateSettings` is called with new value
7. App state updates
8. All components using the setting re-render with new value

### Time Calculations

All time calculations use the configured settings:

```typescript
// Example: Calculating time from slider value
const startMinutes = settings.sleepStartHour * 60;
const totalDurationMinutes = settings.sleepDuration;
const minutesToAdd = Math.round((sliderValue / 100) * totalDurationMinutes);
const currentMinutes = startMinutes + minutesToAdd;
```

## Migration Notes

### Breaking Changes

None. All changes are backward compatible. Default values match previous hardcoded values where applicable.

### Default Values

When the app is first launched, default values are:
- Sleep Start Hour: 22 (10 PM)
- Sleep Duration: 600 minutes (10 hours)
- Fade Out Timer: 5 minutes
- Fade In Timer: 5 minutes
- Signal Duration: 30 minutes

## Future Enhancements

Potential improvements for future versions:

1. **Presets**: Save and load sleep schedule presets
2. **Validation**: Add validation for conflicting settings
3. **Persistence**: Save settings to device storage
4. **Sync**: Sync settings across devices
5. **Advanced Options**: More granular control over fade curves and signal patterns

## Related Files

- `types.ts`: Type definitions
- `App.tsx`: State management
- `screens/SettingsScreen.tsx`: Settings UI
- `screens/HomeScreen.tsx`: Timeline implementation
- `components/SleepChart.tsx`: Chart visualization
- `screens/HistoryScreen.tsx`: History display

