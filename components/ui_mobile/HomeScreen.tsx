/**
 * HomeScreen Example
 * 
 * Demonstrates how to use the useSleepData hook and SleepChart component
 * to display real Apple Health sleep data.
 */

import React from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
  SafeAreaView,
} from 'react-native';
import { useSleepData } from '../hooks/useSleepData';
import { SleepChart } from '../components/SleepChart';

const ACCENT_COLOR = '#92b7c9';

export const HomeScreen: React.FC = () => {
  const {
    lastNight,
    chartData,
    isLoading,
    error,
    hasPermission,
    requestPermission,
    refreshData,
  } = useSleepData();

  // Format duration for display
  const formatDuration = (minutes: number): string => {
    const hours = Math.floor(minutes / 60);
    const mins = Math.round(minutes % 60);
    return `${hours}h ${mins}m`;
  };

  // Format time for display
  const formatTime = (date: Date): string => {
    return date.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView
        contentContainerStyle={styles.scrollContent}
        refreshControl={
          <RefreshControl
            refreshing={isLoading}
            onRefresh={refreshData}
            tintColor={ACCENT_COLOR}
          />
        }
      >
        <Text style={styles.title}>Sleep</Text>

        {/* Permission prompt if needed */}
        {hasPermission === false && (
          <View style={styles.permissionCard}>
            <Text style={styles.permissionTitle}>Enable Health Access</Text>
            <Text style={styles.permissionText}>
              Allow access to Apple Health to see your sleep data.
            </Text>
            <TouchableOpacity
              style={styles.permissionButton}
              onPress={requestPermission}
            >
              <Text style={styles.permissionButtonText}>Grant Access</Text>
            </TouchableOpacity>
          </View>
        )}

        {/* Sleep Chart */}
        <View style={styles.chartCard}>
          <Text style={styles.cardTitle}>Last Night</Text>
          <SleepChart
            accentColor={ACCENT_COLOR}
            isDark={true}
            sleepData={chartData}
            isLoading={isLoading}
            error={error}
          />
        </View>

        {/* Sleep Summary */}
        {lastNight && (
          <View style={styles.summaryCard}>
            <Text style={styles.cardTitle}>Summary</Text>

            <View style={styles.summaryGrid}>
              <View style={styles.summaryItem}>
                <Text style={styles.summaryValue}>
                  {formatDuration(lastNight.summary.totalSleepMinutes)}
                </Text>
                <Text style={styles.summaryLabel}>Total Sleep</Text>
              </View>

              <View style={styles.summaryItem}>
                <Text style={styles.summaryValue}>
                  {Math.round(lastNight.summary.sleepEfficiency)}%
                </Text>
                <Text style={styles.summaryLabel}>Efficiency</Text>
              </View>

              <View style={styles.summaryItem}>
                <Text style={styles.summaryValue}>
                  {formatTime(lastNight.bedTime)}
                </Text>
                <Text style={styles.summaryLabel}>Bedtime</Text>
              </View>

              <View style={styles.summaryItem}>
                <Text style={styles.summaryValue}>
                  {formatTime(lastNight.wakeTime)}
                </Text>
                <Text style={styles.summaryLabel}>Wake Up</Text>
              </View>
            </View>

            {/* Stage breakdown */}
            <View style={styles.stageBreakdown}>
              <Text style={styles.stageTitle}>Sleep Stages</Text>
              
              <View style={styles.stageRow}>
                <View style={[styles.stageDot, { backgroundColor: '#2563eb' }]} />
                <Text style={styles.stageLabel}>Deep</Text>
                <Text style={styles.stageValue}>
                  {formatDuration(lastNight.summary.deepMinutes)}
                </Text>
              </View>

              <View style={styles.stageRow}>
                <View style={[styles.stageDot, { backgroundColor: '#60a5fa' }]} />
                <Text style={styles.stageLabel}>Core</Text>
                <Text style={styles.stageValue}>
                  {formatDuration(lastNight.summary.coreMinutes)}
                </Text>
              </View>

              <View style={styles.stageRow}>
                <View style={[styles.stageDot, { backgroundColor: '#a78bfa' }]} />
                <Text style={styles.stageLabel}>REM</Text>
                <Text style={styles.stageValue}>
                  {formatDuration(lastNight.summary.remMinutes)}
                </Text>
              </View>

              <View style={styles.stageRow}>
                <View style={[styles.stageDot, { backgroundColor: '#f97316' }]} />
                <Text style={styles.stageLabel}>Awake</Text>
                <Text style={styles.stageValue}>
                  {formatDuration(lastNight.summary.awakeMinutes)}
                </Text>
              </View>
            </View>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0f172a',
  },
  scrollContent: {
    padding: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: '700',
    color: '#f8fafc',
    marginBottom: 20,
  },
  chartCard: {
    backgroundColor: '#1e293b',
    borderRadius: 16,
    padding: 16,
    marginBottom: 16,
  },
  summaryCard: {
    backgroundColor: '#1e293b',
    borderRadius: 16,
    padding: 16,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#f8fafc',
    marginBottom: 12,
  },
  summaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
    marginBottom: 20,
  },
  summaryItem: {
    width: '45%',
  },
  summaryValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#f8fafc',
  },
  summaryLabel: {
    fontSize: 13,
    color: '#94a3b8',
    marginTop: 2,
  },
  stageBreakdown: {
    borderTopWidth: 1,
    borderTopColor: '#334155',
    paddingTop: 16,
  },
  stageTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#94a3b8',
    marginBottom: 12,
  },
  stageRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  stageDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    marginRight: 8,
  },
  stageLabel: {
    flex: 1,
    fontSize: 14,
    color: '#cbd5e1',
  },
  stageValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#f8fafc',
  },
  permissionCard: {
    backgroundColor: '#1e293b',
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
    alignItems: 'center',
  },
  permissionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#f8fafc',
    marginBottom: 8,
  },
  permissionText: {
    fontSize: 14,
    color: '#94a3b8',
    textAlign: 'center',
    marginBottom: 16,
  },
  permissionButton: {
    backgroundColor: '#92b7c9',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
  },
  permissionButtonText: {
    color: '#0f172a',
    fontWeight: '600',
    fontSize: 14,
  },
});

export default HomeScreen;
