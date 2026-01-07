# Changelog

## [Unreleased] - Configurable Settings Update

### Added

- **Sleep Schedule Configuration**
  - Sleep Start Hour: Configurable start time for sleep schedule (0-23 hours)
  - Sleep Duration: Configurable sleep window duration (60-1440 minutes)
  - Both settings accessible via Settings → Sleep Schedule section

- **Editable Audio Timers**
  - Fade Out Timer: Now editable (0-10 minutes)
  - Fade In Timer: Now editable (0-10 minutes)
  - Both accessible via Settings → Appearance section

- **Signal Duration Setting**
  - New setting to configure how long sounds play when triggered
  - Accessible via Settings → Appearance section
  - Default: 30 minutes
  - Minimum: 1 minute, no maximum

- **Settings Modals**
  - Consistent modal interface for editing all time-based settings
  - Up/down picker controls for easy value adjustment
  - Cancel and Save actions

### Changed

- **HomeScreen Timeline**
  - Timeline slider now uses configurable sleep schedule instead of hardcoded 10 PM - 8 AM
  - Time labels dynamically generated based on configured schedule
  - All time calculations use settings from UserSettings

- **SleepChart Component**
  - Now accepts `sleepStartHour` and `sleepDuration` as props
  - Chart data points generated dynamically based on configured schedule
  - Time labels adapt to user's sleep schedule
  - Reference lines positioned relative to schedule (40% and 60% marks)

- **Settings Screen**
  - Added "Sleep Schedule" section with two new settings
  - Made Fade Out and Fade In timers editable (previously read-only)
  - Added Signal Duration setting to Appearance section

### Technical Changes

- **Type Definitions** (`types.ts`)
  - Added `sleepStartHour: number` to UserSettings
  - Added `sleepDuration: number` to UserSettings
  - Added `signalDuration: number` to UserSettings
  - Updated comments for fade timers to indicate max 10 minutes

- **Component Props**
  - `HomeScreen`: Now accepts `settings: UserSettings` prop
  - `HistoryScreen`: Now accepts `settings: UserSettings` prop
  - `SleepChart`: Now accepts `sleepStartHour` and `sleepDuration` props

- **State Management**
  - Settings state managed at App level
  - Settings passed down to child components as props
  - Single source of truth for all configuration

### Default Values

- Sleep Start Hour: 22 (10 PM)
- Sleep Duration: 600 minutes (10 hours)
- Fade Out Timer: 5 minutes (changed from 30)
- Fade In Timer: 5 minutes
- Signal Duration: 30 minutes

### Migration

No migration required. All changes are backward compatible with default values matching previous hardcoded values where applicable.

