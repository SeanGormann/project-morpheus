export enum AppTab {
  Settings = 'settings',
  Home = 'home',
  History = 'history',
}

export interface UserSettings {
  darkMode: boolean;
  fadeOutTimer: number; // minutes (max 10)
  fadeInTimer: number; // minutes (max 10)
  signalDuration: number; // minutes
  secondaryColor: string;
  sleepStartHour: number; // 24-hour format (0-23), e.g., 22 for 10 PM
  sleepDuration: number; // minutes, e.g., 600 for 10 hours
  profileImage: string | null; // URI of the user's profile photo
}

export interface JournalEntry {
  id: string;
  text: string;
  timestamp: string;
}

export interface SleepEvent {
  id: string;
  time: string;
  title: string;
  description?: string;
  type: 'sound' | 'snore' | 'stop';
  active?: boolean;
}

export interface NightData {
  date: string;
  day: string;
  totalSleep: string;
  efficiency: number;
  eventsCount: number;
  events: SleepEvent[];
  journal: JournalEntry[];
  isOpen: boolean;
}

// Theme colors for React Native
export const Colors = {
  light: {
    background: '#f8fafc',
    surface: '#ffffff',
    text: '#0f172a',
    textSecondary: '#64748b',
    border: '#e2e8f0',
  },
  dark: {
    background: '#0f172a',
    surface: '#1e293b',
    text: '#ffffff',
    textSecondary: '#94a3b8',
    border: 'rgba(255,255,255,0.1)',
  },
};
