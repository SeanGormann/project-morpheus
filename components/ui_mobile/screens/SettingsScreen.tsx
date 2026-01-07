import React, { useState, useMemo } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  ScrollView, 
  TouchableOpacity, 
  TextInput,
  Image,
  Switch,
  Modal,
  Alert,
  Platform,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { LinearGradient } from 'expo-linear-gradient';
import * as ImagePicker from 'expo-image-picker';
import { UserSettings, Colors } from '../types';
import { WheelPicker } from '../components/WheelPicker';

interface SettingsScreenProps {
  settings: UserSettings;
  updateSettings: (newSettings: Partial<UserSettings>) => void;
  isDark: boolean;
}

export const SettingsScreen: React.FC<SettingsScreenProps> = ({ settings, updateSettings, isDark }) => {
  const colors = isDark ? Colors.dark : Colors.light;
  const [editingSleepStart, setEditingSleepStart] = useState(false);
  const [editingSleepDuration, setEditingSleepDuration] = useState(false);
  const [editingFadeOut, setEditingFadeOut] = useState(false);
  const [editingFadeIn, setEditingFadeIn] = useState(false);
  const [editingSignalDuration, setEditingSignalDuration] = useState(false);
  const [tempSleepStartHour, setTempSleepStartHour] = useState(settings.sleepStartHour);
  const [tempSleepDuration, setTempSleepDuration] = useState(settings.sleepDuration);
  const [tempFadeOutTimer, setTempFadeOutTimer] = useState(settings.fadeOutTimer);
  const [tempFadeInTimer, setTempFadeInTimer] = useState(settings.fadeInTimer);
  const [tempSignalDuration, setTempSignalDuration] = useState(settings.signalDuration);
  
  const colorOptions = [
    { name: 'Sky Blue', value: '#13a4ec' },
    { name: 'Emerald', value: '#10b981' },
    { name: 'Violet', value: '#8b5cf6' },
    { name: 'Rose', value: '#f43f5e' },
  ];

  // Generate picker values
  const hourValues = useMemo(() => 
    Array.from({ length: 24 }, (_, i) => {
      if (i === 0) return '12 AM';
      if (i < 12) return `${i} AM`;
      if (i === 12) return '12 PM';
      return `${i - 12} PM`;
    }), []);

  const durationValues = useMemo(() => 
    Array.from({ length: 47 }, (_, i) => {
      const minutes = (i + 1) * 30;
      const hours = Math.floor(minutes / 60);
      const mins = minutes % 60;
      return `${hours}h ${mins}m`;
    }), []);

  const fadeTimerValues = useMemo(() => 
    Array.from({ length: 11 }, (_, i) => `${i} min`), []);

  const signalDurationValues = useMemo(() => 
    Array.from({ length: 120 }, (_, i) => `${i + 1} min`), []);

  const pickImageFromLibrary = async () => {
    // Request permission
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permission Required',
        'Please allow access to your photo library to change your profile picture.',
        [{ text: 'OK' }]
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ['images'],
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      updateSettings({ profileImage: result.assets[0].uri });
    }
  };

  const takePhoto = async () => {
    // Request permission
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        'Permission Required',
        'Please allow access to your camera to take a profile picture.',
        [{ text: 'OK' }]
      );
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      aspect: [1, 1],
      quality: 0.8,
    });

    if (!result.canceled && result.assets[0]) {
      updateSettings({ profileImage: result.assets[0].uri });
    }
  };

  const showImagePickerOptions = () => {
    if (Platform.OS === 'web') {
      // On web, just use the library picker
      pickImageFromLibrary();
      return;
    }

    Alert.alert(
      'Change Profile Photo',
      'Choose an option',
      [
        { text: 'Take Photo', onPress: takePhoto },
        { text: 'Choose from Library', onPress: pickImageFromLibrary },
        ...(settings.profileImage ? [{ text: 'Remove Photo', onPress: () => updateSettings({ profileImage: null }), style: 'destructive' as const }] : []),
        { text: 'Cancel', style: 'cancel' as const },
      ]
    );
  };

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      {/* Header */}
      <View style={[styles.header, { backgroundColor: colors.background, borderBottomColor: colors.border }]}>
        <TouchableOpacity style={styles.headerButton}>
          <Ionicons name="arrow-back" size={20} color={colors.text} />
        </TouchableOpacity>
        <Text style={[styles.headerTitle, { color: colors.text }]}>Settings</Text>
        <TouchableOpacity style={styles.headerButton}>
          <Text style={[styles.doneButton, { color: settings.secondaryColor }]}>Done</Text>
        </TouchableOpacity>
      </View>

      <ScrollView 
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Profile Section */}
        <View style={styles.profileSection}>
          <TouchableOpacity 
            style={styles.avatarContainer} 
            onPress={showImagePickerOptions}
            activeOpacity={0.8}
          >
            {settings.profileImage ? (
              <Image 
                source={{ uri: settings.profileImage }}
                style={[styles.avatar, { borderColor: colors.surface }]}
              />
            ) : (
              <View style={[styles.avatar, styles.avatarPlaceholder, { borderColor: colors.surface, backgroundColor: colors.border }]}>
                <Ionicons name="person" size={40} color={colors.textSecondary} />
              </View>
            )}
            <View style={[styles.editBadge, { backgroundColor: settings.secondaryColor, borderColor: colors.background }]}>
              <Ionicons name="camera" size={14} color="#fff" />
            </View>
          </TouchableOpacity>
          <Text style={[styles.changePhotoText, { color: settings.secondaryColor }]}>
            Tap to change photo
          </Text>
        </View>

        {/* Account Info */}
        <View style={[styles.card, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <View style={[styles.cardRow, { borderBottomColor: colors.border }]}>
            <Text style={[styles.label, { color: colors.text }]}>Name</Text>
            <TextInput
              style={[styles.input, { color: colors.textSecondary }]}
              defaultValue="Alex Smith"
              placeholderTextColor={colors.textSecondary}
            />
          </View>
          <View style={styles.cardRow}>
            <Text style={[styles.label, { color: colors.text }]}>Email</Text>
            <Text style={[styles.value, { color: colors.textSecondary }]}>alex@example.com</Text>
          </View>
        </View>
        <Text style={[styles.hint, { color: colors.textSecondary }]}>
          This information is used to personalize your sleep analysis.
        </Text>

        {/* Appearance */}
        <Text style={[styles.sectionTitle, { color: colors.textSecondary }]}>Appearance</Text>
        <View style={[styles.card, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          {/* Dark Mode */}
          <View style={[styles.cardRow, { borderBottomColor: colors.border }]}>
            <View style={styles.rowLeft}>
              <View style={[styles.iconContainer, { backgroundColor: 'rgba(99,102,241,0.15)' }]}>
                <Ionicons name="moon" size={18} color="#6366f1" />
              </View>
              <Text style={[styles.label, { color: colors.text }]}>Dark Mode</Text>
            </View>
            <Switch
              value={settings.darkMode}
              onValueChange={(value) => updateSettings({ darkMode: value })}
              trackColor={{ false: '#e2e8f0', true: settings.secondaryColor }}
              thumbColor="#fff"
            />
          </View>

          {/* Fade Out Timer */}
          <TouchableOpacity 
            style={[styles.cardRow, { borderBottomColor: colors.border }]} 
            activeOpacity={0.7}
            onPress={() => {
              setTempFadeOutTimer(settings.fadeOutTimer);
              setEditingFadeOut(true);
            }}
          >
            <View style={styles.rowLeft}>
              <View style={[styles.iconContainer, { backgroundColor: 'rgba(249,115,22,0.15)' }]}>
                <Ionicons name="timer-outline" size={18} color="#f97316" />
              </View>
              <Text style={[styles.label, { color: colors.text }]}>Fade Out Timer</Text>
            </View>
            <View style={styles.rowRight}>
              <Text style={[styles.value, { color: colors.textSecondary }]}>{settings.fadeOutTimer} min</Text>
              <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
            </View>
          </TouchableOpacity>

          {/* Fade In Timer */}
          <TouchableOpacity 
            style={[styles.cardRow, { borderBottomColor: colors.border }]} 
            activeOpacity={0.7}
            onPress={() => {
              setTempFadeInTimer(settings.fadeInTimer);
              setEditingFadeIn(true);
            }}
          >
            <View style={styles.rowLeft}>
              <View style={[styles.iconContainer, { backgroundColor: 'rgba(20,184,166,0.15)' }]}>
                <Ionicons name="timer-outline" size={18} color="#14b8a6" />
              </View>
              <Text style={[styles.label, { color: colors.text }]}>Fade In Timer</Text>
            </View>
            <View style={styles.rowRight}>
              <Text style={[styles.value, { color: colors.textSecondary }]}>{settings.fadeInTimer} min</Text>
              <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
            </View>
          </TouchableOpacity>

          {/* Signal Duration */}
          <TouchableOpacity 
            style={[styles.cardRow, { borderBottomColor: colors.border }]} 
            activeOpacity={0.7}
            onPress={() => {
              setTempSignalDuration(settings.signalDuration);
              setEditingSignalDuration(true);
            }}
          >
            <View style={styles.rowLeft}>
              <View style={[styles.iconContainer, { backgroundColor: 'rgba(236,72,153,0.15)' }]}>
                <Ionicons name="volume-high-outline" size={18} color="#ec4899" />
              </View>
              <Text style={[styles.label, { color: colors.text }]}>Signal Duration</Text>
            </View>
            <View style={styles.rowRight}>
              <Text style={[styles.value, { color: colors.textSecondary }]}>{settings.signalDuration} min</Text>
              <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
            </View>
          </TouchableOpacity>

          {/* Secondary Color */}
          <View style={styles.colorSection}>
            <View style={styles.rowLeft}>
              <View style={[styles.iconContainer, { backgroundColor: 'rgba(236,72,153,0.15)' }]}>
                <Ionicons name="color-palette-outline" size={18} color="#ec4899" />
              </View>
              <Text style={[styles.label, { color: colors.text }]}>Secondary Colour</Text>
            </View>
            <View style={styles.colorPicker}>
              {colorOptions.map((color) => (
                <TouchableOpacity
                  key={color.value}
                  onPress={() => updateSettings({ secondaryColor: color.value })}
                  style={[
                    styles.colorOption,
                    { backgroundColor: color.value },
                    settings.secondaryColor === color.value && styles.colorOptionSelected,
                  ]}
                  activeOpacity={0.7}
                >
                  {settings.secondaryColor === color.value && (
                    <Ionicons name="checkmark" size={14} color="#fff" />
                  )}
                </TouchableOpacity>
              ))}
            </View>
          </View>
        </View>
        <Text style={[styles.hint, { color: colors.textSecondary }]}>
          Reduces eye strain for better sleep preparation.
        </Text>

        {/* Sleep Schedule */}
        <Text style={[styles.sectionTitle, { color: colors.textSecondary }]}>Sleep Schedule</Text>
        <View style={[styles.card, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          {/* Sleep Start Hour */}
          <TouchableOpacity 
            style={[styles.cardRow, { borderBottomColor: colors.border }]} 
            activeOpacity={0.7}
            onPress={() => {
              setTempSleepStartHour(settings.sleepStartHour);
              setEditingSleepStart(true);
            }}
          >
            <View style={styles.rowLeft}>
              <View style={[styles.iconContainer, { backgroundColor: 'rgba(59,130,246,0.15)' }]}>
                <Ionicons name="moon-outline" size={18} color="#3b82f6" />
              </View>
              <Text style={[styles.label, { color: colors.text }]}>Sleep Start Time</Text>
            </View>
            <View style={styles.rowRight}>
              <Text style={[styles.value, { color: colors.textSecondary }]}>
                {settings.sleepStartHour === 0 ? '12 AM' : 
                 settings.sleepStartHour < 12 ? `${settings.sleepStartHour} AM` :
                 settings.sleepStartHour === 12 ? '12 PM' :
                 `${settings.sleepStartHour - 12} PM`}
              </Text>
              <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
            </View>
          </TouchableOpacity>

          {/* Sleep Duration */}
          <TouchableOpacity 
            style={styles.cardRow} 
            activeOpacity={0.7}
            onPress={() => {
              setTempSleepDuration(settings.sleepDuration);
              setEditingSleepDuration(true);
            }}
          >
            <View style={styles.rowLeft}>
              <View style={[styles.iconContainer, { backgroundColor: 'rgba(139,92,246,0.15)' }]}>
                <Ionicons name="time-outline" size={18} color="#8b5cf6" />
              </View>
              <Text style={[styles.label, { color: colors.text }]}>Sleep Duration</Text>
            </View>
            <View style={styles.rowRight}>
              <Text style={[styles.value, { color: colors.textSecondary }]}>
                {Math.floor(settings.sleepDuration / 60)}h {settings.sleepDuration % 60}m
              </Text>
              <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
            </View>
          </TouchableOpacity>
        </View>
        <Text style={[styles.hint, { color: colors.textSecondary }]}>
          Configure your sleep schedule timeline and duration.
        </Text>

        {/* About */}
        <Text style={[styles.sectionTitle, { color: colors.textSecondary }]}>About</Text>
        <View style={[styles.card, { backgroundColor: colors.surface, borderColor: colors.border }]}>
          <View style={[styles.cardRow, { borderBottomColor: colors.border }]}>
            <Text style={[styles.label, { color: colors.text }]}>Version</Text>
            <Text style={[styles.value, { color: colors.textSecondary }]}>1.0.2</Text>
          </View>
          <TouchableOpacity style={styles.cardRow} activeOpacity={0.7}>
            <Text style={[styles.label, { color: colors.text }]}>Privacy Policy</Text>
            <Ionicons name="chevron-forward" size={20} color={colors.textSecondary} />
          </TouchableOpacity>
        </View>

        {/* Logout */}
        <TouchableOpacity 
          style={[styles.logoutButton, { backgroundColor: colors.surface, borderColor: colors.border }]}
          activeOpacity={0.7}
        >
          <Ionicons name="log-out-outline" size={18} color="#ef4444" />
          <Text style={styles.logoutText}>Log Out</Text>
        </TouchableOpacity>

        <Text style={[styles.copyright, { color: colors.textSecondary }]}>
          SleepSounds App © 2024
        </Text>
      </ScrollView>

      {/* Sleep Start Hour Modal */}
      <Modal
        visible={editingSleepStart}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setEditingSleepStart(false)}
      >
        <View style={styles.modalOverlay}>
          <TouchableOpacity 
            style={styles.modalBackdrop} 
            activeOpacity={1} 
            onPress={() => setEditingSleepStart(false)} 
          />
          <View style={[styles.modalContent, { backgroundColor: colors.surface }]}>
            <LinearGradient
              colors={[`${settings.secondaryColor}15`, 'transparent']}
              style={styles.modalGlow}
            />
            <View style={styles.modalHandle} />
            <View style={styles.modalHeader}>
              <View style={[styles.modalIconContainer, { backgroundColor: `${settings.secondaryColor}20` }]}>
                <Ionicons name="moon-outline" size={24} color={settings.secondaryColor} />
              </View>
              <Text style={[styles.modalTitle, { color: colors.text }]}>Sleep Start Time</Text>
              <Text style={[styles.modalSubtitle, { color: colors.textSecondary }]}>
                When your sleep schedule begins
              </Text>
            </View>
            
            <View style={styles.wheelPickerContainer}>
              <WheelPicker
                values={hourValues}
                selectedIndex={tempSleepStartHour}
                onValueChange={(index) => setTempSleepStartHour(index)}
                textColor={colors.textSecondary}
                selectedColor={colors.text}
                backgroundColor={colors.surface}
                accentColor={settings.secondaryColor}
              />
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButtonCancel, { borderColor: colors.border }]}
                onPress={() => setEditingSleepStart(false)}
                activeOpacity={0.7}
              >
                <Text style={[styles.modalButtonText, { color: colors.text }]}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButtonSave, { backgroundColor: settings.secondaryColor }]}
                onPress={() => {
                  updateSettings({ sleepStartHour: tempSleepStartHour });
                  setEditingSleepStart(false);
                }}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={[`${settings.secondaryColor}`, `${settings.secondaryColor}dd`]}
                  style={styles.buttonGradient}
                >
                  <Text style={styles.modalButtonTextPrimary}>Save</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Sleep Duration Modal */}
      <Modal
        visible={editingSleepDuration}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setEditingSleepDuration(false)}
      >
        <View style={styles.modalOverlay}>
          <TouchableOpacity 
            style={styles.modalBackdrop} 
            activeOpacity={1} 
            onPress={() => setEditingSleepDuration(false)} 
          />
          <View style={[styles.modalContent, { backgroundColor: colors.surface }]}>
            <LinearGradient
              colors={[`${settings.secondaryColor}15`, 'transparent']}
              style={styles.modalGlow}
            />
            <View style={styles.modalHandle} />
            <View style={styles.modalHeader}>
              <View style={[styles.modalIconContainer, { backgroundColor: `${settings.secondaryColor}20` }]}>
                <Ionicons name="time-outline" size={24} color={settings.secondaryColor} />
              </View>
              <Text style={[styles.modalTitle, { color: colors.text }]}>Sleep Duration</Text>
              <Text style={[styles.modalSubtitle, { color: colors.textSecondary }]}>
                Total sleep window length
              </Text>
            </View>
            
            <View style={styles.wheelPickerContainer}>
              <WheelPicker
                values={durationValues}
                selectedIndex={Math.round(tempSleepDuration / 30) - 1}
                onValueChange={(index) => setTempSleepDuration((index + 1) * 30)}
                textColor={colors.textSecondary}
                selectedColor={colors.text}
                backgroundColor={colors.surface}
                accentColor={settings.secondaryColor}
              />
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButtonCancel, { borderColor: colors.border }]}
                onPress={() => setEditingSleepDuration(false)}
                activeOpacity={0.7}
              >
                <Text style={[styles.modalButtonText, { color: colors.text }]}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButtonSave, { backgroundColor: settings.secondaryColor }]}
                onPress={() => {
                  updateSettings({ sleepDuration: tempSleepDuration });
                  setEditingSleepDuration(false);
                }}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={[`${settings.secondaryColor}`, `${settings.secondaryColor}dd`]}
                  style={styles.buttonGradient}
                >
                  <Text style={styles.modalButtonTextPrimary}>Save</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Fade Out Timer Modal */}
      <Modal
        visible={editingFadeOut}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setEditingFadeOut(false)}
      >
        <View style={styles.modalOverlay}>
          <TouchableOpacity 
            style={styles.modalBackdrop} 
            activeOpacity={1} 
            onPress={() => setEditingFadeOut(false)} 
          />
          <View style={[styles.modalContent, { backgroundColor: colors.surface }]}>
            <LinearGradient
              colors={[`${settings.secondaryColor}15`, 'transparent']}
              style={styles.modalGlow}
            />
            <View style={styles.modalHandle} />
            <View style={styles.modalHeader}>
              <View style={[styles.modalIconContainer, { backgroundColor: 'rgba(249,115,22,0.15)' }]}>
                <Ionicons name="volume-low-outline" size={24} color="#f97316" />
              </View>
              <Text style={[styles.modalTitle, { color: colors.text }]}>Fade Out Timer</Text>
              <Text style={[styles.modalSubtitle, { color: colors.textSecondary }]}>
                Gradually decrease volume over time
              </Text>
            </View>
            
            <View style={styles.wheelPickerContainer}>
              <WheelPicker
                values={fadeTimerValues}
                selectedIndex={tempFadeOutTimer}
                onValueChange={(index) => setTempFadeOutTimer(index)}
                textColor={colors.textSecondary}
                selectedColor={colors.text}
                backgroundColor={colors.surface}
                accentColor="#f97316"
              />
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButtonCancel, { borderColor: colors.border }]}
                onPress={() => setEditingFadeOut(false)}
                activeOpacity={0.7}
              >
                <Text style={[styles.modalButtonText, { color: colors.text }]}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButtonSave, { backgroundColor: '#f97316' }]}
                onPress={() => {
                  updateSettings({ fadeOutTimer: tempFadeOutTimer });
                  setEditingFadeOut(false);
                }}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={['#f97316', '#ea580c']}
                  style={styles.buttonGradient}
                >
                  <Text style={styles.modalButtonTextPrimary}>Save</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Fade In Timer Modal */}
      <Modal
        visible={editingFadeIn}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setEditingFadeIn(false)}
      >
        <View style={styles.modalOverlay}>
          <TouchableOpacity 
            style={styles.modalBackdrop} 
            activeOpacity={1} 
            onPress={() => setEditingFadeIn(false)} 
          />
          <View style={[styles.modalContent, { backgroundColor: colors.surface }]}>
            <LinearGradient
              colors={[`${settings.secondaryColor}15`, 'transparent']}
              style={styles.modalGlow}
            />
            <View style={styles.modalHandle} />
            <View style={styles.modalHeader}>
              <View style={[styles.modalIconContainer, { backgroundColor: 'rgba(20,184,166,0.15)' }]}>
                <Ionicons name="volume-high-outline" size={24} color="#14b8a6" />
              </View>
              <Text style={[styles.modalTitle, { color: colors.text }]}>Fade In Timer</Text>
              <Text style={[styles.modalSubtitle, { color: colors.textSecondary }]}>
                Gradually increase volume on start
              </Text>
            </View>
            
            <View style={styles.wheelPickerContainer}>
              <WheelPicker
                values={fadeTimerValues}
                selectedIndex={tempFadeInTimer}
                onValueChange={(index) => setTempFadeInTimer(index)}
                textColor={colors.textSecondary}
                selectedColor={colors.text}
                backgroundColor={colors.surface}
                accentColor="#14b8a6"
              />
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButtonCancel, { borderColor: colors.border }]}
                onPress={() => setEditingFadeIn(false)}
                activeOpacity={0.7}
              >
                <Text style={[styles.modalButtonText, { color: colors.text }]}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButtonSave, { backgroundColor: '#14b8a6' }]}
                onPress={() => {
                  updateSettings({ fadeInTimer: tempFadeInTimer });
                  setEditingFadeIn(false);
                }}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={['#14b8a6', '#0d9488']}
                  style={styles.buttonGradient}
                >
                  <Text style={styles.modalButtonTextPrimary}>Save</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Signal Duration Modal */}
      <Modal
        visible={editingSignalDuration}
        transparent={true}
        animationType="fade"
        onRequestClose={() => setEditingSignalDuration(false)}
      >
        <View style={styles.modalOverlay}>
          <TouchableOpacity 
            style={styles.modalBackdrop} 
            activeOpacity={1} 
            onPress={() => setEditingSignalDuration(false)} 
          />
          <View style={[styles.modalContent, { backgroundColor: colors.surface }]}>
            <LinearGradient
              colors={[`${settings.secondaryColor}15`, 'transparent']}
              style={styles.modalGlow}
            />
            <View style={styles.modalHandle} />
            <View style={styles.modalHeader}>
              <View style={[styles.modalIconContainer, { backgroundColor: 'rgba(236,72,153,0.15)' }]}>
                <Ionicons name="musical-notes-outline" size={24} color="#ec4899" />
              </View>
              <Text style={[styles.modalTitle, { color: colors.text }]}>Signal Duration</Text>
              <Text style={[styles.modalSubtitle, { color: colors.textSecondary }]}>
                How long each sound signal plays
              </Text>
            </View>
            
            <View style={styles.wheelPickerContainer}>
              <WheelPicker
                values={signalDurationValues}
                selectedIndex={tempSignalDuration - 1}
                onValueChange={(index) => setTempSignalDuration(index + 1)}
                textColor={colors.textSecondary}
                selectedColor={colors.text}
                backgroundColor={colors.surface}
                accentColor="#ec4899"
              />
            </View>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={[styles.modalButtonCancel, { borderColor: colors.border }]}
                onPress={() => setEditingSignalDuration(false)}
                activeOpacity={0.7}
              >
                <Text style={[styles.modalButtonText, { color: colors.text }]}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalButtonSave, { backgroundColor: '#ec4899' }]}
                onPress={() => {
                  updateSettings({ signalDuration: tempSignalDuration });
                  setEditingSignalDuration(false);
                }}
                activeOpacity={0.8}
              >
                <LinearGradient
                  colors={['#ec4899', '#db2777']}
                  style={styles.buttonGradient}
                >
                  <Text style={styles.modalButtonTextPrimary}>Save</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
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
    borderBottomWidth: 1,
  },
  headerButton: {
    padding: 8,
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '700',
  },
  doneButton: {
    fontSize: 16,
    fontWeight: '500',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    padding: 16,
    paddingBottom: 100,
  },
  profileSection: {
    alignItems: 'center',
    paddingVertical: 16,
  },
  avatarContainer: {
    position: 'relative',
  },
  avatar: {
    width: 96,
    height: 96,
    borderRadius: 48,
    borderWidth: 4,
  },
  avatarPlaceholder: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  changePhotoText: {
    marginTop: 12,
    fontSize: 14,
    fontWeight: '500',
  },
  editBadge: {
    position: 'absolute',
    bottom: 0,
    right: 0,
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 2,
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    borderRadius: 12,
    borderWidth: 1,
    overflow: 'hidden',
  },
  cardRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 14,
    minHeight: 56,
    borderBottomWidth: 1,
  },
  rowLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  rowRight: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  iconContainer: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  label: {
    fontSize: 16,
    fontWeight: '500',
  },
  input: {
    flex: 1,
    textAlign: 'right',
    fontSize: 16,
  },
  value: {
    fontSize: 14,
  },
  hint: {
    fontSize: 12,
    marginTop: 8,
    marginBottom: 24,
    paddingHorizontal: 16,
  },
  sectionTitle: {
    fontSize: 13,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    paddingHorizontal: 16,
    marginBottom: 8,
  },
  colorSection: {
    paddingHorizontal: 16,
    paddingVertical: 14,
    gap: 12,
  },
  colorPicker: {
    flexDirection: 'row',
    gap: 8,
    marginLeft: 44,
    marginTop: 8,
  },
  colorOption: {
    width: 32,
    height: 32,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  colorOptionSelected: {
    borderWidth: 2,
    borderColor: 'rgba(255,255,255,0.5)',
    transform: [{ scale: 1.1 }],
  },
  logoutButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    paddingVertical: 14,
    borderRadius: 12,
    borderWidth: 1,
    marginTop: 16,
  },
  logoutText: {
    color: '#ef4444',
    fontSize: 16,
    fontWeight: '500',
  },
  copyright: {
    textAlign: 'center',
    fontSize: 12,
    fontWeight: '500',
    marginTop: 24,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.6)',
    justifyContent: 'flex-end',
  },
  modalBackdrop: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
  },
  modalContent: {
    borderTopLeftRadius: 32,
    borderTopRightRadius: 32,
    paddingHorizontal: 24,
    paddingBottom: 40,
    paddingTop: 12,
    overflow: 'hidden',
  },
  modalGlow: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 150,
  },
  modalHandle: {
    width: 40,
    height: 4,
    backgroundColor: 'rgba(255,255,255,0.2)',
    borderRadius: 2,
    alignSelf: 'center',
    marginBottom: 20,
  },
  modalHeader: {
    alignItems: 'center',
    marginBottom: 8,
  },
  modalIconContainer: {
    width: 56,
    height: 56,
    borderRadius: 28,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: '700',
    marginBottom: 8,
    textAlign: 'center',
    letterSpacing: -0.5,
  },
  modalSubtitle: {
    fontSize: 15,
    textAlign: 'center',
    opacity: 0.8,
  },
  wheelPickerContainer: {
    marginVertical: 16,
  },
  modalButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 8,
  },
  modalButtonCancel: {
    flex: 1,
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: 'center',
    borderWidth: 1.5,
  },
  modalButtonSave: {
    flex: 1.5,
    borderRadius: 16,
    overflow: 'hidden',
  },
  buttonGradient: {
    paddingVertical: 16,
    alignItems: 'center',
    justifyContent: 'center',
  },
  modalButtonText: {
    fontSize: 16,
    fontWeight: '600',
  },
  modalButtonTextPrimary: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '700',
    letterSpacing: 0.3,
  },
});
