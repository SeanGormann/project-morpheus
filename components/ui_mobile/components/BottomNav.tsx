import React from 'react';
import { View, TouchableOpacity, Text, StyleSheet } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { AppTab, Colors } from '../types';

interface BottomNavProps {
  activeTab: AppTab;
  onTabChange: (tab: AppTab) => void;
  accentColor: string;
  isDark: boolean;
}

export const BottomNav: React.FC<BottomNavProps> = ({ activeTab, onTabChange, accentColor, isDark }) => {
  const colors = isDark ? Colors.dark : Colors.light;
  const inactiveColor = '#94a3b8';

  const getIconColor = (tab: AppTab) => activeTab === tab ? accentColor : inactiveColor;

  return (
    <View style={[styles.container, { 
      backgroundColor: isDark ? 'rgba(30,41,59,0.95)' : 'rgba(255,255,255,0.95)',
      borderTopColor: colors.border,
    }]}>
      <TouchableOpacity 
        style={styles.tab}
        onPress={() => onTabChange(AppTab.Settings)}
        activeOpacity={0.7}
      >
        <Ionicons name="settings-outline" size={24} color={getIconColor(AppTab.Settings)} />
        <Text style={[styles.label, { color: getIconColor(AppTab.Settings) }]}>Settings</Text>
      </TouchableOpacity>

      <TouchableOpacity 
        style={styles.tab}
        onPress={() => onTabChange(AppTab.Home)}
        activeOpacity={0.7}
      >
        <Ionicons name="home-outline" size={24} color={getIconColor(AppTab.Home)} />
        <Text style={[styles.label, { color: getIconColor(AppTab.Home) }]}>Home</Text>
      </TouchableOpacity>

      <TouchableOpacity 
        style={styles.tab}
        onPress={() => onTabChange(AppTab.History)}
        activeOpacity={0.7}
      >
        <Ionicons name="time-outline" size={24} color={getIconColor(AppTab.History)} />
        <Text style={[styles.label, { color: getIconColor(AppTab.History) }]}>History</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    height: 70,
    paddingTop: 12,
    paddingBottom: 12,
    borderTopWidth: 1,
  },
  tab: {
    alignItems: 'center',
    gap: 4,
    padding: 8,
  },
  label: {
    fontSize: 10,
    fontWeight: '500',
  },
});
