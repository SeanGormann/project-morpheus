# Documentation

This directory contains documentation for the SleepSounds mobile app.

## Available Documentation

- **[Configurable Settings](./configurable-settings.md)**: Comprehensive guide to the configurable settings system, including sleep schedule, audio timers, and signal duration settings.

- **[Changelog](./CHANGELOG.md)**: Detailed changelog of recent updates, including the configurable settings feature.

## Quick Reference

### Settings Overview

The app supports the following configurable settings:

**Sleep Schedule**:
- Sleep Start Hour (0-23)
- Sleep Duration (60-1440 minutes)

**Audio Settings**:
- Fade Out Timer (0-10 minutes)
- Fade In Timer (0-10 minutes)
- Signal Duration (1+ minutes)

**Appearance**:
- Dark Mode (toggle)
- Secondary Color (color picker)

### Key Files

- `types.ts`: Type definitions for UserSettings
- `App.tsx`: Settings state management
- `screens/SettingsScreen.tsx`: Settings UI and modals
- `screens/HomeScreen.tsx`: Timeline implementation using settings
- `components/SleepChart.tsx`: Chart visualization using settings

## Contributing

When adding new settings or modifying existing ones:

1. Update `UserSettings` interface in `types.ts`
2. Add default value in `App.tsx`
3. Add UI in `SettingsScreen.tsx`
4. Update components that use the setting
5. Update this documentation

