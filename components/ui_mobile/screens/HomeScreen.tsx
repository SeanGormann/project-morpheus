import React, { useState, useRef, useEffect } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  ScrollView, 
  TouchableOpacity,
  Dimensions,
  Animated,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import Slider from '@react-native-community/slider';
import { Colors, UserSettings } from '../types';

interface HomeScreenProps {
  accentColor: string;
  isDark: boolean;
  settings: UserSettings;
}

interface ScheduledEvent {
  id: number;
  timeDisplay: string;
  value: number;
}

export const HomeScreen: React.FC<HomeScreenProps> = ({ accentColor, isDark, settings }) => {
  const colors = isDark ? Colors.dark : Colors.light;
  const [selectedTime, setSelectedTime] = useState(45);
  const [isPlaying, setIsPlaying] = useState(false);
  const [volume, setVolume] = useState(65);
  const [events, setEvents] = useState<ScheduledEvent[]>([
    { id: 1, timeDisplay: "02:00 AM", value: 40 },
    { id: 2, timeDisplay: "04:30 AM", value: 65 }
  ]);
  const [showHelp, setShowHelp] = useState(false);
  const [isLocked, setIsLocked] = useState(false);
  
  // Precision mode state
  const [isPrecisionMode, setIsPrecisionMode] = useState(false);
  const [precisionCenter, setPrecisionCenter] = useState(45); // Center point when precision mode activates
  const [isSliding, setIsSliding] = useState(false);
  const lastValueRef = useRef(selectedTime);
  const holdTimerRef = useRef<NodeJS.Timeout | null>(null);
  const precisionAnimValue = useRef(new Animated.Value(0)).current;
  
  // Precision mode window: ±30 minutes = 60 minutes total
  // In a 600-min (10 hour) schedule, 60 minutes = 10% of the slider
  const PRECISION_WINDOW_MINUTES = 60;
  const precisionRange = (PRECISION_WINDOW_MINUTES / settings.sleepDuration) * 100;
  
  // Clear timer on unmount
  useEffect(() => {
    return () => {
      if (holdTimerRef.current) clearTimeout(holdTimerRef.current);
    };
  }, []);
  
  // Animate precision mode transitions
  useEffect(() => {
    Animated.spring(precisionAnimValue, {
      toValue: isPrecisionMode ? 1 : 0,
      useNativeDriver: false,
      tension: 80,
      friction: 10,
    }).start();
  }, [isPrecisionMode]);
  
  const startHoldTimer = () => {
    if (holdTimerRef.current) clearTimeout(holdTimerRef.current);
    holdTimerRef.current = setTimeout(() => {
      if (isSliding && !isPrecisionMode) {
        // Enter precision mode
        setPrecisionCenter(selectedTime);
        setIsPrecisionMode(true);
      }
    }, 2000);
  };
  
  const handleSliderStart = () => {
    setIsSliding(true);
    lastValueRef.current = selectedTime;
    startHoldTimer();
  };
  
  const handleSliderChange = (value: number) => {
    // If value changed significantly, restart the hold timer
    if (Math.abs(value - lastValueRef.current) > 0.5) {
      lastValueRef.current = value;
      startHoldTimer();
    }
    
    if (isPrecisionMode) {
      // In precision mode, map the 0-100 slider to the precision window
      const halfRange = precisionRange / 2;
      const minBound = Math.max(0, precisionCenter - halfRange);
      const maxBound = Math.min(100, precisionCenter + halfRange);
      const actualValue = minBound + (value / 100) * (maxBound - minBound);
      setSelectedTime(actualValue);
    } else {
      setSelectedTime(value);
    }
  };
  
  const handleSliderEnd = () => {
    setIsSliding(false);
    if (holdTimerRef.current) {
      clearTimeout(holdTimerRef.current);
      holdTimerRef.current = null;
    }
    // Exit precision mode after a short delay to allow final adjustments
    if (isPrecisionMode) {
      setTimeout(() => {
        setIsPrecisionMode(false);
      }, 300);
    }
  };
  
  // Get the slider value to display (inverse mapping for precision mode)
  const getSliderDisplayValue = () => {
    if (!isPrecisionMode) return selectedTime;
    
    const halfRange = precisionRange / 2;
    const minBound = Math.max(0, precisionCenter - halfRange);
    const maxBound = Math.min(100, precisionCenter + halfRange);
    return ((selectedTime - minBound) / (maxBound - minBound)) * 100;
  };

  const calculateTime = (val: number) => {
    const startMinutes = settings.sleepStartHour * 60;
    const totalDurationMinutes = settings.sleepDuration;
    const minutesToAdd = Math.round((val / 100) * totalDurationMinutes);
    
    let currentMinutes = startMinutes + minutesToAdd;
    let hours = Math.floor(currentMinutes / 60) % 24;
    let minutes = currentMinutes % 60;
    
    const ampm = hours >= 12 ? 'PM' : 'AM';
    const displayHours = hours % 12 || 12;
    const displayMinutes = minutes < 10 ? `0${minutes}` : minutes;
    
    return {
      formatted: `${displayHours}:${displayMinutes}`,
      ampm: ampm,
      full: `${displayHours}:${displayMinutes} ${ampm}`
    };
  };

  const timeData = calculateTime(selectedTime);

  const handleAddEvent = () => {
    if (isLocked) return;
    const newEvent: ScheduledEvent = {
      id: Date.now(),
      timeDisplay: timeData.full,
      value: selectedTime
    };
    setEvents(prev => [...prev, newEvent].sort((a, b) => a.value - b.value));
  };

  const handleRemoveEvent = (id: number) => {
    if (isLocked) return;
    setEvents(prev => prev.filter(e => e.id !== id));
  };

  const toggleLock = () => {
    setIsLocked(!isLocked);
  };

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      {/* Header */}
      <View style={[styles.header, { backgroundColor: colors.background }]}>
        <View style={styles.headerSpacer} />
        <Text style={[styles.headerTitle, { color: colors.text }]}>Morpheus Console</Text>
        <TouchableOpacity 
          onPress={() => setShowHelp(true)}
          style={styles.helpButton}
        >
          <Text style={[styles.helpText, { color: accentColor }]}>Help</Text>
        </TouchableOpacity>
      </View>

      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Title Section */}
        <View style={styles.section}>
          <Text style={[styles.sectionTitle, { color: colors.text }]}>Configure</Text>
          <Text style={[styles.sectionSubtitle, { color: colors.textSecondary }]}>
            Slide to select a time & add it to your schedule. Hold the slider still for 2s to zoom in for precision.
          </Text>
        </View>

        {/* Time Display */}
        <View style={styles.timeDisplay}>
          <View style={styles.timeLabelRow}>
            <Text style={[styles.timeLabel, { color: colors.textSecondary }]}>Selected Time</Text>
            {isPrecisionMode && (
              <Animated.View 
                style={[
                  styles.precisionBadge, 
                  { 
                    backgroundColor: `${accentColor}20`,
                    borderColor: accentColor,
                    opacity: precisionAnimValue,
                    transform: [{
                      scale: precisionAnimValue.interpolate({
                        inputRange: [0, 1],
                        outputRange: [0.8, 1],
                      })
                    }]
                  }
                ]}
              >
                <Ionicons name="search" size={12} color={accentColor} />
                <Text style={[styles.precisionBadgeText, { color: accentColor }]}>PRECISION</Text>
              </Animated.View>
            )}
          </View>
          <View style={styles.timeRow}>
            <Text style={[
              styles.timeValue, 
              { color: isLocked ? colors.textSecondary : colors.text }
            ]}>
              {timeData.formatted}
            </Text>
            <Text style={[styles.timeAmPm, { color: colors.textSecondary }]}>
              {timeData.ampm}
            </Text>
          </View>
        </View>

        {/* Timeline Slider */}
        <Animated.View style={[
          styles.sliderContainer, 
          { 
            backgroundColor: isDark ? '#1e293b' : '#e2e8f0',
            opacity: isLocked ? 0.6 : 1,
            borderWidth: precisionAnimValue.interpolate({
              inputRange: [0, 1],
              outputRange: [0, 2],
            }),
            borderColor: accentColor,
            transform: [{
              scaleY: precisionAnimValue.interpolate({
                inputRange: [0, 1],
                outputRange: [1, 1.15],
              })
            }]
          }
        ]}>
          {/* Precision mode glow effect */}
          {isPrecisionMode && (
            <LinearGradient
              colors={[`${accentColor}30`, 'transparent', `${accentColor}30`]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
              style={styles.precisionGlow}
            />
          )}
          
          {/* Marker dots - hide in precision mode */}
          {!isPrecisionMode && (
            <>
              <View style={[styles.markerLine, { left: '35%', backgroundColor: `${accentColor}40` }]} />
              <View style={[styles.markerDot, { left: '35%', backgroundColor: accentColor }]} />
              <View style={[styles.markerLine, { left: '65%', backgroundColor: `${accentColor}40` }]} />
              <View style={[styles.markerDot, { left: '65%', backgroundColor: accentColor }]} />
            </>
          )}
          
          {/* Precision mode minute markers */}
          {isPrecisionMode && (
            <View style={styles.precisionMarkers}>
              {Array.from({ length: 13 }, (_, i) => (
                <View
                  key={i}
                  style={[
                    styles.precisionTick,
                    {
                      backgroundColor: i === 6 ? accentColor : `${accentColor}40`,
                      height: i % 3 === 0 ? 24 : 12,
                    }
                  ]}
                />
              ))}
            </View>
          )}
          
          <Slider
            style={styles.slider}
            minimumValue={0}
            maximumValue={100}
            step={isPrecisionMode ? 0.167 : 0.5}
            value={getSliderDisplayValue()}
            onValueChange={handleSliderChange}
            onSlidingStart={handleSliderStart}
            onSlidingComplete={handleSliderEnd}
            minimumTrackTintColor={accentColor}
            maximumTrackTintColor={isDark ? '#334155' : '#cbd5e1'}
            thumbTintColor={accentColor}
            disabled={isLocked}
          />
        </Animated.View>

        {/* Time Labels */}
        <View style={styles.timeLabels}>
          {(() => {
            if (isPrecisionMode) {
              // Show minute-level labels in precision mode
              const halfRange = precisionRange / 2;
              const minBound = Math.max(0, precisionCenter - halfRange);
              const maxBound = Math.min(100, precisionCenter + halfRange);
              
              const labels = [];
              const numLabels = 5;
              
              for (let i = 0; i < numLabels; i++) {
                const progress = i / (numLabels - 1);
                const sliderVal = minBound + progress * (maxBound - minBound);
                const timeInfo = calculateTime(sliderVal);
                labels.push(timeInfo.full);
              }
              
              return labels.map((label, idx) => (
                <Text key={idx} style={[styles.timeSliderLabel, styles.precisionLabel, { color: accentColor }]}>
                  {label}
                </Text>
              ));
            }
            
            // Normal mode labels
            const labels = [];
            const startHour = settings.sleepStartHour;
            const duration = settings.sleepDuration;
            const numLabels = 6;
            
            for (let i = 0; i < numLabels; i++) {
              const progress = i / (numLabels - 1);
              const totalMinutes = startHour * 60 + (progress * duration);
              const hours = Math.floor(totalMinutes / 60) % 24;
              const minutes = totalMinutes % 60;
              
              // Round to nearest hour for cleaner labels
              const roundedHours = minutes >= 30 ? (hours + 1) % 24 : hours;
              const ampm = roundedHours >= 12 ? 'PM' : 'AM';
              const displayHours = roundedHours % 12 || 12;
              const label = `${displayHours} ${ampm}`;
              
              labels.push(label);
            }
            
            return labels.map((label, idx) => (
              <Text key={idx} style={[styles.timeSliderLabel, { color: colors.textSecondary }]}>
                {label}
              </Text>
            ));
          })()}
        </View>
        
        {/* Precision mode hint */}
        <Animated.View style={[
          styles.precisionHint,
          {
            opacity: precisionAnimValue.interpolate({
              inputRange: [0, 1],
              outputRange: [0, 1],
            }),
            transform: [{
              translateY: precisionAnimValue.interpolate({
                inputRange: [0, 1],
                outputRange: [10, 0],
              })
            }]
          }
        ]}>
          <Ionicons name="finger-print-outline" size={14} color={colors.textSecondary} />
          <Text style={[styles.precisionHintText, { color: colors.textSecondary }]}>
            Minute-level precision active
          </Text>
        </Animated.View>

        {/* Add Button */}
        <TouchableOpacity
          style={[
            styles.addButton,
            { 
              backgroundColor: colors.surface,
              borderColor: colors.border,
              opacity: isLocked ? 0.5 : 1,
            }
          ]}
          onPress={handleAddEvent}
          disabled={isLocked}
          activeOpacity={0.7}
        >
          <Ionicons 
            name="add-circle-outline" 
            size={20} 
            color={isLocked ? colors.textSecondary : accentColor} 
          />
          <Text style={[styles.addButtonText, { color: colors.text }]}>
            {isLocked ? 'Timeline Locked' : `Add ${timeData.full}`}
          </Text>
        </TouchableOpacity>

        {/* Scheduled Events */}
        <View style={styles.eventsSection}>
          <Text style={[styles.eventsLabel, { color: colors.textSecondary }]}>
            Scheduled Events
          </Text>
          <View style={styles.eventsList}>
            {events.length === 0 && (
              <Text style={[styles.noEvents, { color: colors.textSecondary }]}>
                No events scheduled.
              </Text>
            )}
            {events.map((event) => (
              <View 
                key={event.id}
                style={[
                  styles.eventChip,
                  { 
                    backgroundColor: `${accentColor}15`,
                    borderColor: `${accentColor}30`,
                    opacity: isLocked ? 0.7 : 1,
                  }
                ]}
              >
                <Text style={[styles.eventTime, { color: accentColor }]}>
                  {event.timeDisplay}
                </Text>
                {!isLocked && (
                  <TouchableOpacity 
                    onPress={() => handleRemoveEvent(event.id)}
                    style={styles.removeEventBtn}
                  >
                    <Ionicons name="close" size={14} color={`${accentColor}80`} />
                  </TouchableOpacity>
                )}
              </View>
            ))}
          </View>
        </View>

        {/* Sound Check Panel */}
        <View style={styles.soundCheckSection}>
          <View style={styles.soundCheckHeader}>
            <Text style={[styles.soundCheckTitle, { color: colors.text }]}>Sound Check</Text>
            <View style={styles.visualizer}>
              {[1, 2, 3, 4, 5].map((i) => (
                <View 
                  key={i} 
                  style={[
                    styles.visualizerBar,
                    { 
                      backgroundColor: accentColor,
                      height: isPlaying ? 8 + Math.random() * 12 : 4,
                    }
                  ]} 
                />
              ))}
            </View>
          </View>

          <View style={[
            styles.soundCheckPanel, 
            { 
              backgroundColor: isDark ? '#1c2e36' : '#f1f5f9',
              borderColor: colors.border,
            }
          ]}>
            <View style={styles.soundCheckControls}>
              <TouchableOpacity 
                onPress={() => setIsPlaying(!isPlaying)}
                style={[styles.playButton, { backgroundColor: accentColor }]}
                activeOpacity={0.8}
              >
                <Ionicons 
                  name={isPlaying ? 'pause' : 'play'} 
                  size={28} 
                  color="#fff" 
                  style={isPlaying ? {} : { marginLeft: 4 }}
                />
              </TouchableOpacity>

              <View style={styles.volumeControl}>
                <View style={styles.volumeHeader}>
                  <Text style={[styles.volumeLabel, { color: colors.textSecondary }]}>Volume</Text>
                  <Text style={[styles.volumeValue, { color: colors.textSecondary }]}>{volume}%</Text>
                </View>
                <View style={styles.volumeSliderRow}>
                  <Ionicons name="volume-mute" size={18} color={colors.textSecondary} />
                  <Slider
                    style={styles.volumeSlider}
                    minimumValue={0}
                    maximumValue={100}
                    value={volume}
                    onValueChange={setVolume}
                    minimumTrackTintColor={accentColor}
                    maximumTrackTintColor={isDark ? '#475569' : '#cbd5e1'}
                    thumbTintColor={accentColor}
                  />
                  <Ionicons name="volume-high" size={18} color={colors.textSecondary} />
                </View>
              </View>
            </View>

            <View style={[styles.soundCheckInfo, { borderTopColor: colors.border }]}>
              <Ionicons name="pulse" size={20} color={accentColor} />
              <Text style={[styles.soundCheckInfoText, { color: colors.textSecondary }]}>
                Testing: 40Hz Binaural Beat (Pure Tone)
              </Text>
            </View>
          </View>
        </View>

        {/* Lock Button */}
        <View style={[styles.lockButtonContainer, { backgroundColor: colors.background }]}>
          <TouchableOpacity
            onPress={toggleLock}
            style={[
              styles.lockButton,
              { backgroundColor: isLocked ? '#475569' : accentColor }
            ]}
            activeOpacity={0.9}
          >
            <Ionicons 
              name={isLocked ? 'create-outline' : 'lock-closed'} 
              size={20} 
              color="#fff" 
            />
            <Text style={styles.lockButtonText}>
              {isLocked ? 'Edit Schedule' : 'Lock-in Schedule'}
            </Text>
          </TouchableOpacity>
        </View>
      </ScrollView>

      {/* Help Modal */}
      {showHelp && (
        <View style={styles.helpOverlay}>
          <View style={[styles.helpModal, { backgroundColor: colors.surface }]}>
            <TouchableOpacity 
              onPress={() => setShowHelp(false)}
              style={[styles.helpCloseBtn, { backgroundColor: isDark ? 'rgba(255,255,255,0.1)' : '#f1f5f9' }]}
            >
              <Ionicons name="close" size={20} color={colors.text} />
            </TouchableOpacity>

            <View style={[styles.helpHeader, { borderBottomColor: colors.border }]}>
              <View style={[styles.helpIconContainer, { backgroundColor: `${accentColor}20` }]}>
                <Ionicons name="help-circle" size={24} color={accentColor} />
              </View>
              <Text style={[styles.helpTitle, { color: colors.text }]}>Quick Guide</Text>
            </View>

            <View style={styles.helpContent}>
              <View style={styles.helpItem}>
                <View style={[styles.helpItemIcon, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f8fafc' }]}>
                  <Ionicons name="time-outline" size={20} color={colors.textSecondary} />
                </View>
                <View style={styles.helpItemText}>
                  <Text style={[styles.helpItemTitle, { color: colors.text }]}>Set Your Schedule</Text>
                  <Text style={[styles.helpItemDesc, { color: colors.textSecondary }]}>
                    Use the slider to pick a time. Hold still for 2 seconds to activate precision mode for minute-level control.
                  </Text>
                </View>
              </View>

              <View style={styles.helpItem}>
                <View style={[styles.helpItemIcon, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f8fafc' }]}>
                  <Ionicons name="play-outline" size={20} color={colors.textSecondary} />
                </View>
                <View style={styles.helpItemText}>
                  <Text style={[styles.helpItemTitle, { color: colors.text }]}>Sound Check</Text>
                  <Text style={[styles.helpItemDesc, { color: colors.textSecondary }]}>
                    Tap the play button to test the audio volume.
                  </Text>
                </View>
              </View>

              <View style={styles.helpItem}>
                <View style={[styles.helpItemIcon, { backgroundColor: isDark ? 'rgba(255,255,255,0.05)' : '#f8fafc' }]}>
                  <Ionicons name="bar-chart-outline" size={20} color={colors.textSecondary} />
                </View>
                <View style={styles.helpItemText}>
                  <Text style={[styles.helpItemTitle, { color: colors.text }]}>Track Progress</Text>
                  <Text style={[styles.helpItemDesc, { color: colors.textSecondary }]}>
                    Visit the History tab to view sleep stats and journal.
                  </Text>
                </View>
              </View>
            </View>

            <TouchableOpacity 
              onPress={() => setShowHelp(false)}
              style={[styles.helpGotItBtn, { backgroundColor: accentColor }]}
              activeOpacity={0.9}
            >
              <Text style={styles.helpGotItText}>Got it</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
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
  helpButton: {
    paddingHorizontal: 8,
    paddingVertical: 8,
  },
  helpText: {
    fontSize: 16,
    fontWeight: '700',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingTop: 8,
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 28,
    fontWeight: '700',
    marginBottom: 8,
  },
  sectionSubtitle: {
    fontSize: 16,
    lineHeight: 24,
  },
  timeDisplay: {
    alignItems: 'center',
    marginBottom: 16,
  },
  timeLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  timeLabel: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  precisionBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    paddingHorizontal: 8,
    paddingVertical: 3,
    borderRadius: 12,
    borderWidth: 1,
  },
  precisionBadgeText: {
    fontSize: 9,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  timeRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
  },
  timeValue: {
    fontSize: 48,
    fontWeight: '700',
    fontVariant: ['tabular-nums'],
  },
  timeAmPm: {
    fontSize: 20,
    fontWeight: '500',
    marginLeft: 8,
  },
  sliderContainer: {
    height: 100,
    borderRadius: 16,
    justifyContent: 'center',
    position: 'relative',
    overflow: 'hidden',
  },
  precisionGlow: {
    ...StyleSheet.absoluteFillObject,
  },
  markerLine: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    width: 2,
  },
  markerDot: {
    position: 'absolute',
    top: '50%',
    width: 12,
    height: 12,
    borderRadius: 6,
    marginTop: -6,
    marginLeft: -6,
  },
  precisionMarkers: {
    position: 'absolute',
    top: 0,
    bottom: 0,
    left: 20,
    right: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  precisionTick: {
    width: 1,
    borderRadius: 0.5,
  },
  slider: {
    width: '100%',
    height: 40,
  },
  timeLabels: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 4,
    marginTop: 8,
    marginBottom: 16,
  },
  timeSliderLabel: {
    fontSize: 10,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  precisionLabel: {
    fontSize: 9,
    fontWeight: '600',
  },
  precisionHint: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 6,
    marginTop: -8,
    marginBottom: 16,
  },
  precisionHintText: {
    fontSize: 12,
    fontWeight: '500',
  },
  addButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    borderRadius: 12,
    borderWidth: 1,
  },
  addButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  eventsSection: {
    marginTop: 16,
    minHeight: 80,
  },
  eventsLabel: {
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: 12,
    marginLeft: 4,
  },
  eventsList: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  noEvents: {
    fontSize: 14,
    fontStyle: 'italic',
    marginLeft: 4,
  },
  eventChip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingLeft: 12,
    paddingRight: 8,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
  },
  eventTime: {
    fontSize: 14,
    fontWeight: '600',
  },
  removeEventBtn: {
    padding: 4,
  },
  soundCheckSection: {
    marginTop: 24,
  },
  soundCheckHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  soundCheckTitle: {
    fontSize: 18,
    fontWeight: '700',
  },
  visualizer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 2,
    height: 20,
  },
  visualizerBar: {
    width: 3,
    borderRadius: 2,
  },
  soundCheckPanel: {
    borderRadius: 16,
    padding: 20,
    borderWidth: 1,
  },
  soundCheckControls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  playButton: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
  },
  volumeControl: {
    flex: 1,
  },
  volumeHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  volumeLabel: {
    fontSize: 12,
    fontWeight: '500',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  volumeValue: {
    fontSize: 12,
    fontWeight: '500',
  },
  volumeSliderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  volumeSlider: {
    flex: 1,
    height: 40,
  },
  soundCheckInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 16,
    paddingTop: 16,
    borderTopWidth: 1,
  },
  soundCheckInfoText: {
    fontSize: 14,
  },
  lockButtonContainer: {
    padding: 16,
    paddingTop: 24,
    paddingBottom: 16,
  },
  lockButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 16,
    borderRadius: 12,
  },
  lockButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '700',
  },
  helpOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  helpModal: {
    width: '100%',
    maxWidth: 340,
    borderRadius: 24,
    padding: 24,
  },
  helpCloseBtn: {
    position: 'absolute',
    top: 16,
    right: 16,
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
  },
  helpHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingBottom: 16,
    borderBottomWidth: 1,
    marginBottom: 24,
  },
  helpIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    alignItems: 'center',
    justifyContent: 'center',
  },
  helpTitle: {
    fontSize: 20,
    fontWeight: '700',
  },
  helpContent: {
    gap: 20,
    marginBottom: 24,
  },
  helpItem: {
    flexDirection: 'row',
    gap: 12,
  },
  helpItemIcon: {
    width: 40,
    height: 40,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  helpItemText: {
    flex: 1,
  },
  helpItemTitle: {
    fontSize: 15,
    fontWeight: '600',
    marginBottom: 4,
  },
  helpItemDesc: {
    fontSize: 14,
    lineHeight: 20,
  },
  helpGotItBtn: {
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
  },
  helpGotItText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
  },
});
