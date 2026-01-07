# SleepSounds - Expo React Native App

A sleep tracking and sound scheduling mobile app built with Expo and React Native.

## Features

- **Sound Console**: Configure your nightly sound schedule with an intuitive timeline slider
- **History**: Track your sleep patterns and efficiency over time
- **Journal**: Record private notes and audio memos about your sleep
- **Settings**: Customize dark mode, timers, and accent colors

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- Expo CLI
- iOS Simulator (Mac) or Android Emulator

### Installation

```bash
# Install dependencies
npm install

# Start the development server
npx expo start
```

### Running on iOS Simulator

```bash
npx expo start --ios
```

### Running on Android Emulator

```bash
npx expo start --android
```

### Running on Physical Device

1. Install the **Expo Go** app on your device
2. Run `npx expo start`
3. Scan the QR code with:
   - iOS: Camera app
   - Android: Expo Go app

## Project Structure

```
ui_mobile/
├── App.tsx              # Main app component with navigation
├── index.tsx            # Entry point
├── types.ts             # TypeScript types and theme colors
├── components/
│   ├── BottomNav.tsx    # Tab navigation component
│   └── SleepChart.tsx   # Sleep cycle visualization
├── screens/
│   ├── HomeScreen.tsx   # Sound console and scheduling
│   ├── HistoryScreen.tsx # Sleep history and journal
│   └── SettingsScreen.tsx # App settings
└── assets/              # App icons and images
```

## Tech Stack

- **Expo** (~54.0.0)
- **React Native** (0.76.5)
- **TypeScript** (~5.3.3)
- **React Native SVG** (for charts)
- **@expo/vector-icons** (Ionicons)
