import React, { useState, useEffect, useMemo } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  ScrollView, 
  TouchableOpacity,
  TextInput,
  RefreshControl,
  ActivityIndicator,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { NightData, Colors, UserSettings } from '../types';
import { SleepChart } from '../components/SleepChart';
import { useSleepData } from '../hooks/useSleepData';
import { SleepSession, SleepChartDataPoint } from '../types/health';
import { healthKitService } from '../services/healthKitService';

interface HistoryScreenProps {
  accentColor: string;
  isDark: boolean;
  settings: UserSettings;
}

// Helper to format duration
const formatDuration = (minutes: number): string => {
  const hours = Math.floor(minutes / 60);
  const mins = Math.round(minutes % 60);
  return `${hours}h ${mins}m`;
};

// Helper to format date for display
const formatDay = (date: Date): string => {
  return date.toLocaleDateString('en-US', {
    weekday: 'long',
    month: 'short',
    day: 'numeric',
  });
};

// Helper to format date as YYYY-MM-DD
const formatDateId = (date: Date): string => {
  return date.toISOString().split('T')[0];
};

// Convert SleepSession to NightData format
const sessionToNightData = (session: SleepSession): NightData => ({
  date: formatDateId(session.date),
  day: formatDay(session.date),
  totalSleep: formatDuration(session.summary.totalSleepMinutes),
  efficiency: Math.round(session.summary.sleepEfficiency),
  eventsCount: 0,
  isOpen: false,
  events: [],
  journal: [],
});

export const HistoryScreen: React.FC<HistoryScreenProps> = ({ accentColor, isDark, settings }) => {
  const colors = isDark ? Colors.dark : Colors.light;
  
  // HealthKit data
  const {
    history,
    isLoading,
    isLoadingMore,
    error,
    hasPermission,
    hasMoreData,
    loadedDays,
    requestPermission,
    refreshData,
    loadMore,
  } = useSleepData();

  // Convert HealthKit sessions to NightData and merge with local journal data
  const [localJournalData, setLocalJournalData] = useState<Record<string, NightData['journal']>>({});
  
  const historyData = useMemo(() => {
    if (history.length === 0) {
      // Return empty array when no HealthKit data
      return [];
    }
    
    return history.map((session) => {
      const nightData = sessionToNightData(session);
      // Merge any locally stored journal entries
      if (localJournalData[nightData.date]) {
        nightData.journal = localJournalData[nightData.date];
      }
      return nightData;
    });
  }, [history, localJournalData]);

  const [selectedNightIndex, setSelectedNightIndex] = useState<number | null>(null);
  
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);

  // Get chart data for selected night
  const selectedChartData = useMemo((): SleepChartDataPoint[] | null => {
    if (selectedNightIndex === null || !history[selectedNightIndex]) {
      return null;
    }
    return healthKitService.sessionToChartData(history[selectedNightIndex]);
  }, [selectedNightIndex, history]);

  // Get selected session for summary display
  const selectedSession = useMemo((): SleepSession | null => {
    if (selectedNightIndex === null || !history[selectedNightIndex]) {
      return null;
    }
    return history[selectedNightIndex];
  }, [selectedNightIndex, history]);

  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isRecording) {
      interval = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    } else {
      setRecordingTime(0);
    }
    return () => clearInterval(interval);
  }, [isRecording]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins < 10 ? '0' : ''}${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  // Calendar helpers
  const today = new Date();
  const getCalendarDays = () => {
    const days = [];
    for (let i = -3; i <= 3; i++) {
      const date = new Date(today);
      date.setDate(date.getDate() + i);
      days.push(date);
    }
    return days;
  };
  const calendarDays = getCalendarDays();
  const daysOfWeek = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

  const handleAddNote = () => {
    if (selectedNightIndex === null || !historyData[selectedNightIndex]) return;
    
    const nightDate = historyData[selectedNightIndex].date;
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    const newEntry = {
      id: Date.now().toString(),
      text: '',
      timestamp: timeString
    };
    
    setLocalJournalData(prev => ({
      ...prev,
      [nightDate]: [...(prev[nightDate] || []), newEntry]
    }));
  };

  const handleUpdateNote = (entryId: string, text: string) => {
    if (selectedNightIndex === null || !historyData[selectedNightIndex]) return;
    
    const nightDate = historyData[selectedNightIndex].date;
    
    setLocalJournalData(prev => ({
      ...prev,
      [nightDate]: (prev[nightDate] || []).map(entry =>
        entry.id === entryId ? { ...entry, text } : entry
      )
    }));
  };

  const handleDeleteNote = (entryId: string) => {
    if (selectedNightIndex === null || !historyData[selectedNightIndex]) return;
    
    const nightDate = historyData[selectedNightIndex].date;
    
    setLocalJournalData(prev => ({
      ...prev,
      [nightDate]: (prev[nightDate] || []).filter(entry => entry.id !== entryId)
    }));
  };

  const handleStartRecording = () => {
    setIsRecording(true);
  };

  const handleStopRecording = () => {
    setIsRecording(false);
    if (selectedNightIndex === null || !historyData[selectedNightIndex]) return;

    const nightDate = historyData[selectedNightIndex].date;
    const now = new Date();
    const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    
    const newEntry = {
      id: Date.now().toString(),
      text: '🎤 Audio Note (0:15)',
      timestamp: timeString
    };
    
    setLocalJournalData(prev => ({
      ...prev,
      [nightDate]: [...(prev[nightDate] || []), newEntry]
    }));
  };

  // Permission prompt view
  if (hasPermission === false) {
    return (
      <View style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={[styles.listHeader, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
          <View style={styles.headerSpacer} />
          <Text style={[styles.headerTitle, { color: colors.text }]}>History</Text>
          <View style={styles.headerSpacer} />
        </View>
        <View style={styles.permissionCard}>
          <Ionicons name="heart" size={48} color={accentColor} />
          <Text style={[styles.permissionTitle, { color: colors.text }]}>Enable Health Access</Text>
          <Text style={[styles.permissionText, { color: colors.textSecondary }]}>
            Allow access to Apple Health to see your real sleep data from your Apple Watch.
          </Text>
          <TouchableOpacity
            style={[styles.permissionButton, { backgroundColor: accentColor }]}
            onPress={requestPermission}
          >
            <Text style={styles.permissionButtonText}>Grant Access</Text>
          </TouchableOpacity>
        </View>
      </View>
    );
  }

  // Detail View
  if (selectedNightIndex !== null && historyData[selectedNightIndex]) {
    const night = historyData[selectedNightIndex];
    const journalEntries = localJournalData[night.date] || [];
    
    return (
      <View style={[styles.container, { backgroundColor: colors.background }]}>
        {/* Header */}
        <View style={[styles.detailHeader, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
          <TouchableOpacity 
            onPress={() => setSelectedNightIndex(null)}
            style={styles.backButton}
          >
            <Ionicons name="arrow-back" size={20} color={colors.text} />
          </TouchableOpacity>
          <Text style={[styles.headerTitle, { color: colors.text }]}>{night.day}</Text>
          <View style={styles.headerSpacer} />
        </View>

        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          {/* Stats */}
          <View style={styles.statsGrid}>
            <View style={[styles.statCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
              <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Total Sleep</Text>
              <View style={styles.statValueRow}>
                <Text style={[styles.statValue, { color: colors.text }]}>{night.totalSleep}</Text>
              </View>
            </View>
            <View style={[styles.statCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
              <Text style={[styles.statLabel, { color: colors.textSecondary }]}>Time Asleep</Text>
              <View style={styles.statValueRow}>
                <Text style={[styles.statValue, { color: colors.text }]}>{night.efficiency}%</Text>
                <Text style={[styles.statNote, { color: colors.textSecondary }]}>
                  of time in bed
                </Text>
              </View>
            </View>
          </View>

          {/* Sleep Stages Summary */}
          {selectedSession && (
            <View style={[styles.stagesCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
              <Text style={[styles.chartTitle, { color: colors.text }]}>Sleep Stages</Text>
              <View style={styles.stagesGrid}>
                <View style={styles.stageItem}>
                  <View style={[styles.stageDot, { backgroundColor: '#2563eb' }]} />
                  <Text style={[styles.stageLabel, { color: colors.textSecondary }]}>Deep</Text>
                  <Text style={[styles.stageValue, { color: colors.text }]}>
                    {formatDuration(selectedSession.summary.deepMinutes)}
                  </Text>
                </View>
                <View style={styles.stageItem}>
                  <View style={[styles.stageDot, { backgroundColor: '#60a5fa' }]} />
                  <Text style={[styles.stageLabel, { color: colors.textSecondary }]}>Core</Text>
                  <Text style={[styles.stageValue, { color: colors.text }]}>
                    {formatDuration(selectedSession.summary.coreMinutes)}
                  </Text>
                </View>
                <View style={styles.stageItem}>
                  <View style={[styles.stageDot, { backgroundColor: '#a78bfa' }]} />
                  <Text style={[styles.stageLabel, { color: colors.textSecondary }]}>REM</Text>
                  <Text style={[styles.stageValue, { color: colors.text }]}>
                    {formatDuration(selectedSession.summary.remMinutes)}
                  </Text>
                </View>
                <View style={styles.stageItem}>
                  <View style={[styles.stageDot, { backgroundColor: '#f97316' }]} />
                  <Text style={[styles.stageLabel, { color: colors.textSecondary }]}>Awake</Text>
                  <Text style={[styles.stageValue, { color: colors.text }]}>
                    {formatDuration(selectedSession.summary.awakeMinutes)}
                  </Text>
                </View>
              </View>
            </View>
          )}

          {/* Chart */}
          <View style={[styles.chartCard, { backgroundColor: colors.surface, borderColor: colors.border }]}>
            <View style={styles.chartHeader}>
              <Text style={[styles.chartTitle, { color: colors.text }]}>Sleep Cycle</Text>
            </View>
            <SleepChart 
              accentColor={accentColor} 
              isDark={isDark}
              sleepData={selectedChartData}
            />
          </View>

          {/* Journal */}
          <View style={styles.journalSection}>
            <View style={styles.journalHeader}>
              <View style={styles.journalTitleRow}>
                <Ionicons name="lock-closed" size={16} color={colors.textSecondary} />
                <Text style={[styles.journalTitle, { color: colors.text }]}>Journal</Text>
              </View>
              <Text style={[styles.journalSubtitle, { color: colors.textSecondary }]}>Private Notes</Text>
            </View>

            {journalEntries.map((entry) => (
              <View key={entry.id} style={styles.journalEntry}>
                {entry.text.startsWith('🎤') ? (
                  <View style={[styles.audioNote, { backgroundColor: colors.surface, borderColor: colors.border }]}>
                    <TouchableOpacity style={styles.audioPlayBtn}>
                      <Ionicons name="play-circle" size={32} color={accentColor} />
                    </TouchableOpacity>
                    <View style={styles.audioInfo}>
                      <Text style={[styles.audioTitle, { color: colors.text }]}>Audio Note</Text>
                      <Text style={[styles.audioMeta, { color: colors.textSecondary }]}>0:15 • {entry.timestamp}</Text>
                    </View>
                    <View style={styles.waveform}>
                      {[4, 8, 3, 6, 9, 5, 2, 7, 4, 6].map((h, i) => (
                        <View 
                          key={i} 
                          style={[styles.waveBar, { height: h * 4, backgroundColor: colors.textSecondary }]} 
                        />
                      ))}
                    </View>
                    <TouchableOpacity 
                      onPress={() => handleDeleteNote(entry.id)}
                      style={styles.deleteNoteBtn}
                    >
                      <Ionicons name="trash-outline" size={16} color={colors.textSecondary} />
                    </TouchableOpacity>
                  </View>
                ) : (
                  <View style={styles.textNoteContainer}>
                    <TextInput
                      value={entry.text}
                      onChangeText={(text) => handleUpdateNote(entry.id, text)}
                      placeholder="Write your thoughts here..."
                      placeholderTextColor={colors.textSecondary}
                      multiline
                      style={[
                        styles.textNote, 
                        { 
                          backgroundColor: colors.surface, 
                          borderColor: colors.border,
                          color: colors.text,
                        }
                      ]}
                    />
                    <Text style={[styles.noteTimestamp, { color: colors.textSecondary }]}>{entry.timestamp}</Text>
                    <TouchableOpacity 
                      onPress={() => handleDeleteNote(entry.id)}
                      style={styles.deleteNoteBtn}
                    >
                      <Ionicons name="trash-outline" size={16} color={colors.textSecondary} />
                    </TouchableOpacity>
                  </View>
                )}
              </View>
            ))}

            <View style={styles.journalActions}>
              <TouchableOpacity 
                onPress={handleAddNote}
                style={[styles.addNoteBtn, { borderColor: colors.border }]}
                activeOpacity={0.7}
              >
                <Ionicons name="add" size={20} color={colors.textSecondary} />
                <Text style={[styles.addNoteText, { color: colors.textSecondary }]}>Add Note</Text>
              </TouchableOpacity>
              <TouchableOpacity 
                onPress={handleStartRecording}
                style={[styles.recordBtn, { backgroundColor: colors.surface, borderColor: colors.border }]}
                activeOpacity={0.7}
              >
                <Ionicons name="mic" size={20} color={accentColor} />
              </TouchableOpacity>
            </View>
          </View>

          {/* Recording Overlay */}
          {isRecording && (
            <View style={[styles.recordingOverlay, { backgroundColor: colors.surface, borderColor: colors.border }]}>
              <View style={[styles.recordingIndicator, { backgroundColor: 'rgba(239,68,68,0.1)' }]}>
                <Ionicons name="mic" size={24} color="#ef4444" />
              </View>
              <View style={styles.recordingInfo}>
                <Text style={[styles.recordingText, { color: colors.text }]}>Recording...</Text>
                <Text style={styles.recordingTime}>{formatTime(recordingTime)}</Text>
              </View>
              <View style={styles.recordingWave}>
                {[1, 2, 3, 4, 5, 6].map(i => (
                  <View 
                    key={i} 
                    style={[styles.recordingBar, { backgroundColor: '#ef4444' }]} 
                  />
                ))}
              </View>
              <TouchableOpacity 
                onPress={handleStopRecording}
                style={[styles.stopBtn, { backgroundColor: isDark ? 'rgba(255,255,255,0.1)' : '#f1f5f9' }]}
              >
                <Ionicons name="stop-circle" size={24} color={colors.text} />
              </TouchableOpacity>
            </View>
          )}
        </ScrollView>
      </View>
    );
  }

  // List View
  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      {/* Header */}
      <View style={[styles.listHeader, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <View style={styles.headerSpacer} />
        <Text style={[styles.headerTitle, { color: colors.text }]}>History</Text>
        <View style={styles.headerSpacer} />
      </View>

      {/* Calendar Strip */}
      <View style={[styles.calendarSection, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <View style={styles.calendarNav}>
          <TouchableOpacity>
            <Ionicons name="chevron-back" size={20} color={colors.textSecondary} />
          </TouchableOpacity>
          <Text style={[styles.calendarMonth, { color: colors.text }]}>
            {today.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
          </Text>
          <TouchableOpacity>
            <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
          </TouchableOpacity>
        </View>
        <View style={styles.calendarDays}>
          {calendarDays.map((date, i) => {
            const isToday = date.toDateString() === today.toDateString();
            return (
              <TouchableOpacity key={date.toISOString()} style={styles.calendarDay}>
                <Text style={[
                  styles.dayLabel,
                  { color: isToday ? accentColor : colors.textSecondary }
                ]}>
                  {daysOfWeek[date.getDay()]}
                </Text>
                <View style={[
                  styles.dayNumber,
                  isToday && { backgroundColor: accentColor }
                ]}>
                  <Text style={[
                    styles.dayNumberText,
                    { color: isToday ? '#fff' : colors.textSecondary }
                  ]}>
                    {date.getDate()}
                  </Text>
                </View>
              </TouchableOpacity>
            );
          })}
        </View>
      </View>

      {/* Loading state */}
      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator color={accentColor} size="large" />
          <Text style={[styles.loadingText, { color: colors.textSecondary }]}>
            Loading sleep data...
          </Text>
        </View>
      )}

      {/* Error state */}
      {error && !isLoading && (
        <View style={styles.errorContainer}>
          <Ionicons name="alert-circle" size={48} color="#ef4444" />
          <Text style={[styles.errorText, { color: colors.text }]}>
            Unable to load sleep data
          </Text>
          <Text style={[styles.errorSubtext, { color: colors.textSecondary }]}>
            {error}
          </Text>
          <TouchableOpacity
            style={[styles.retryButton, { backgroundColor: accentColor }]}
            onPress={refreshData}
          >
            <Text style={styles.retryButtonText}>Try Again</Text>
          </TouchableOpacity>
        </View>
      )}

      {/* Empty state */}
      {!isLoading && !error && historyData.length === 0 && (
        <View style={styles.emptyContainer}>
          <Ionicons name="moon-outline" size={48} color={colors.textSecondary} />
          <Text style={[styles.emptyTitle, { color: colors.text }]}>
            No sleep data yet
          </Text>
          <Text style={[styles.emptySubtext, { color: colors.textSecondary }]}>
            Wear your Apple Watch to bed to track your sleep. Data will appear here automatically.
          </Text>
        </View>
      )}

      {/* Night List */}
      {!isLoading && !error && historyData.length > 0 && (
        <ScrollView 
          style={styles.scrollView}
          contentContainerStyle={styles.nightList}
          showsVerticalScrollIndicator={false}
          refreshControl={
            <RefreshControl
              refreshing={isLoading}
              onRefresh={refreshData}
              tintColor={accentColor}
            />
          }
        >
          {historyData.map((night, index) => (
            <TouchableOpacity 
              key={night.date}
              onPress={() => setSelectedNightIndex(index)}
              style={[styles.nightCard, { backgroundColor: colors.surface, borderColor: colors.border }]}
              activeOpacity={0.7}
            >
              <View style={styles.nightInfo}>
                <Text style={[styles.nightDay, { color: colors.text }]}>{night.day}</Text>
                <View style={styles.nightMeta}>
                  <Ionicons name="moon" size={14} color={colors.textSecondary} />
                  <Text style={[styles.nightMetaText, { color: colors.textSecondary }]}>
                    {night.totalSleep} Sleep
                  </Text>
                  <View style={[styles.metaDot, { backgroundColor: colors.textSecondary }]} />
                  <Text style={[styles.nightMetaText, { color: colors.textSecondary }]}>
                    {night.efficiency}% asleep
                  </Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
            </TouchableOpacity>
          ))}

          {/* Load More Button */}
          {hasMoreData && (
            <TouchableOpacity
              onPress={loadMore}
              disabled={isLoadingMore}
              style={[styles.loadMoreButton, { borderColor: colors.border }]}
              activeOpacity={0.7}
            >
              {isLoadingMore ? (
                <ActivityIndicator size="small" color={accentColor} />
              ) : (
                <>
                  <Ionicons name="time-outline" size={18} color={colors.textSecondary} />
                  <Text style={[styles.loadMoreText, { color: colors.textSecondary }]}>
                    Load More ({loadedDays} days loaded)
                  </Text>
                </>
              )}
            </TouchableOpacity>
          )}
        </ScrollView>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 100,
  },
  listHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  detailHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderBottomWidth: 1,
  },
  headerSpacer: {
    width: 40,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
    flex: 1,
    textAlign: 'center',
  },
  backButton: {
    width: 40,
    height: 40,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 20,
  },
  calendarSection: {
    paddingHorizontal: 16,
    paddingBottom: 16,
    borderBottomWidth: 1,
  },
  calendarNav: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 16,
  },
  calendarMonth: {
    fontSize: 14,
    fontWeight: '700',
  },
  calendarDays: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  calendarDay: {
    alignItems: 'center',
    gap: 4,
  },
  dayLabel: {
    fontSize: 10,
    fontWeight: '700',
    textTransform: 'uppercase',
  },
  dayNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dayNumberText: {
    fontSize: 14,
    fontWeight: '500',
  },
  nightList: {
    padding: 16,
    gap: 12,
  },
  nightCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 12,
  },
  nightInfo: {
    gap: 4,
  },
  nightDay: {
    fontSize: 16,
    fontWeight: '700',
  },
  nightMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  nightMetaText: {
    fontSize: 14,
  },
  metaDot: {
    width: 4,
    height: 4,
    borderRadius: 2,
  },
  loadMoreButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderStyle: 'dashed',
    marginTop: 4,
    marginBottom: 24,
  },
  loadMoreText: {
    fontSize: 14,
    fontWeight: '500',
  },
  statsGrid: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  statCard: {
    flex: 1,
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
  },
  statLabel: {
    fontSize: 11,
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 4,
  },
  statValueRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    gap: 6,
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
  },
  statBadge: {
    fontSize: 12,
    fontWeight: '500',
    color: '#10b981',
  },
  statNote: {
    fontSize: 12,
    fontWeight: '500',
  },
  stagesCard: {
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 16,
  },
  stagesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginTop: 12,
  },
  stageItem: {
    width: '45%',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  stageDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  stageLabel: {
    fontSize: 13,
    flex: 1,
  },
  stageValue: {
    fontSize: 13,
    fontWeight: '600',
  },
  chartCard: {
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    marginBottom: 24,
  },
  chartHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  chartTitle: {
    fontSize: 14,
    fontWeight: '600',
  },
  chartLegend: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legendText: {
    fontSize: 12,
  },
  journalSection: {
    gap: 16,
  },
  journalHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  journalTitleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  journalTitle: {
    fontSize: 18,
    fontWeight: '700',
  },
  journalSubtitle: {
    fontSize: 11,
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  journalEntry: {
    marginBottom: 12,
  },
  audioNote: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    gap: 12,
  },
  audioPlayBtn: {},
  audioInfo: {},
  audioTitle: {
    fontSize: 14,
    fontWeight: '500',
  },
  audioMeta: {
    fontSize: 12,
  },
  waveform: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    marginLeft: 16,
    opacity: 0.5,
  },
  waveBar: {
    width: 3,
    borderRadius: 2,
  },
  textNoteContainer: {
    position: 'relative',
  },
  textNote: {
    minHeight: 100,
    borderRadius: 12,
    borderWidth: 1,
    padding: 16,
    fontSize: 14,
    textAlignVertical: 'top',
  },
  noteTimestamp: {
    position: 'absolute',
    bottom: 12,
    right: 16,
    fontSize: 10,
    fontWeight: '500',
  },
  deleteNoteBtn: {
    position: 'absolute',
    top: 12,
    right: 12,
    padding: 4,
  },
  journalActions: {
    flexDirection: 'row',
    gap: 12,
  },
  addNoteBtn: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderStyle: 'dashed',
  },
  addNoteText: {
    fontSize: 15,
    fontWeight: '500',
  },
  recordBtn: {
    width: 56,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 12,
    borderWidth: 1,
  },
  recordingOverlay: {
    position: 'absolute',
    left: 16,
    right: 16,
    bottom: 16,
    flexDirection: 'row',
    alignItems: 'center',
    padding: 16,
    borderRadius: 16,
    borderWidth: 1,
    gap: 12,
  },
  recordingIndicator: {
    width: 48,
    height: 48,
    borderRadius: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  recordingInfo: {
    gap: 2,
  },
  recordingText: {
    fontSize: 14,
    fontWeight: '700',
  },
  recordingTime: {
    fontSize: 12,
    fontFamily: 'monospace',
    color: '#ef4444',
  },
  recordingWave: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    marginRight: 8,
  },
  recordingBar: {
    width: 3,
    height: 16,
    borderRadius: 2,
  },
  stopBtn: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  // Permission, Loading, Error, Empty states
  permissionCard: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    gap: 16,
  },
  permissionTitle: {
    fontSize: 20,
    fontWeight: '700',
  },
  permissionText: {
    fontSize: 15,
    textAlign: 'center',
    lineHeight: 22,
  },
  permissionButton: {
    paddingHorizontal: 24,
    paddingVertical: 14,
    borderRadius: 12,
    marginTop: 8,
  },
  permissionButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  loadingContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
  },
  loadingText: {
    fontSize: 15,
  },
  errorContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    gap: 12,
  },
  errorText: {
    fontSize: 18,
    fontWeight: '600',
  },
  errorSubtext: {
    fontSize: 14,
    textAlign: 'center',
  },
  retryButton: {
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
    marginTop: 8,
  },
  retryButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
  },
  emptyContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    gap: 12,
  },
  emptyTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  emptySubtext: {
    fontSize: 14,
    textAlign: 'center',
    lineHeight: 20,
  },
});
