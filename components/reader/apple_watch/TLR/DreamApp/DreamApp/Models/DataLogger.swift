//
//  DataLogger.swift
//  DreamApp
//
//  Project Morpheus — Phase 1: Local data capture
//

import Foundation

/// Logs epoch data from Watch for inspection and export.
/// No server, no model — just capture what we receive.
class DataLogger {
    
    struct EpochData: Codable {
        let epochIndex: Int
        let timestamp: Date
        let hrFeature: Double
        let heartRatesCount: Int
        let heartRatesAvg: Double
        let heartRates: [Double]
        let motionRowsCount: Int
        let motionSampleFirst: [Double]?
        let motionSampleLast: [Double]?
    }
    
    struct SessionExport: Codable {
        let sessionStart: Date
        let sessionEnd: Date
        let epochCount: Int
        let epochs: [EpochData]
    }
    
    private var epochs: [EpochData] = []
    private var sessionStart: Date?
    
    var epochCount: Int { epochs.count }
    
    func startSession() {
        sessionStart = Date()
        epochs.removeAll()
    }
    
    func log(
        epochIndex: Int,
        hrFeature: Double,
        heartRates: [Double],
        motionData: [[Double]]
    ) {
        let avgHR = heartRates.isEmpty ? 0 : heartRates.reduce(0, +) / Double(heartRates.count)
        let firstMotion = motionData.first
        let lastMotion = motionData.last
        
        let epoch = EpochData(
            epochIndex: epochIndex,
            timestamp: Date(),
            hrFeature: hrFeature,
            heartRatesCount: heartRates.count,
            heartRatesAvg: avgHR,
            heartRates: heartRates,
            motionRowsCount: motionData.count,
            motionSampleFirst: firstMotion,
            motionSampleLast: lastMotion
        )
        epochs.append(epoch)
    }
    
    /// Export session to JSON file. Returns URL for sharing, or nil if no data.
    func exportSession() -> URL? {
        guard let start = sessionStart, !epochs.isEmpty else { return nil }
        
        let session = SessionExport(
            sessionStart: start,
            sessionEnd: Date(),
            epochCount: epochs.count,
            epochs: epochs
        )
        
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        guard let data = try? encoder.encode(session) else { return nil }
        
        let filename = "morpheus_session_\(Int(Date().timeIntervalSince1970)).json"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        
        do {
            try data.write(to: url)
            return url
        } catch {
            print("DataLogger: Failed to write export: \(error)")
            return nil
        }
    }
    
    func reset() {
        epochs.removeAll()
        sessionStart = nil
    }
}
