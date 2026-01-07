import React, { useState } from 'react';
import { View, StyleSheet, useColorScheme } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { BottomNav } from './components/BottomNav';
import { SettingsScreen } from './screens/SettingsScreen';
import { HomeScreen } from './screens/HomeScreen';
import { HistoryScreen } from './screens/HistoryScreen';
import { AppTab, UserSettings, Colors } from './types';

const App: React.FC = () => {
  const systemColorScheme = useColorScheme();
  const [activeTab, setActiveTab] = useState<AppTab>(AppTab.Home);
  
  const [settings, setSettings] = useState<UserSettings>({
    darkMode: systemColorScheme === 'dark',
    fadeOutTimer: 5,
    fadeInTimer: 5,
    signalDuration: 30,
    secondaryColor: '#13a4ec',
    sleepStartHour: 22, // 10 PM
    sleepDuration: 600, // 10 hours
    profileImage: null,
  });

  const updateSettings = (newSettings: Partial<UserSettings>) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  };

  const isDark = settings.darkMode;
  const colors = isDark ? Colors.dark : Colors.light;

  const renderScreen = () => {
    switch (activeTab) {
      case AppTab.Settings:
        return <SettingsScreen settings={settings} updateSettings={updateSettings} isDark={isDark} />;
      case AppTab.History:
        return <HistoryScreen accentColor={settings.secondaryColor} isDark={isDark} settings={settings} />;
      case AppTab.Home:
      default:
        return <HomeScreen accentColor={settings.secondaryColor} isDark={isDark} settings={settings} />;
    }
  };

  return (
    <SafeAreaProvider>
      <StatusBar style={isDark ? 'light' : 'dark'} />
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]} edges={['top']}>
        <View style={[styles.content, { backgroundColor: colors.background }]}>
          {renderScreen()}
        </View>
        <BottomNav 
          activeTab={activeTab} 
          onTabChange={setActiveTab} 
          accentColor={settings.secondaryColor}
          isDark={isDark}
        />
      </SafeAreaView>
    </SafeAreaProvider>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
  },
});

export default App;
