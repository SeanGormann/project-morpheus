//
//  SummaryView.swift
//  DreamApp
//
//  Project Morpheus — Phase 1: Export session
//

import SwiftUI

struct SummaryView: View {
    
    @ObservedObject var sessionManager: ServerCommunication
    @State private var showShareSheet = false
    @State private var exportURL: URL?
    
    var body: some View {
        VStack(spacing: 24) {
            Text("Epochs captured: \(sessionManager.epochCount)")
                .font(.title2)
                .padding()
            
            Text("Sound stimuli was activated \(sessionManager.numberOfTimesActivated) times")
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            Button {
                exportURL = sessionManager.exportSession()
                if exportURL != nil {
                    showShareSheet = true
                }
            } label: {
                Label("Export Session (JSON)", systemImage: "square.and.arrow.up")
                    .padding()
            }
            .disabled(sessionManager.epochCount == 0)
        }
        .padding()
        .onAppear(perform: start)
        .sheet(isPresented: $showShareSheet) {
            if let url = exportURL {
                ShareSheet(items: [url])
            }
        }
    }
    
    func start() {
        sessionManager.endServer()
    }
}

/// Share sheet for exporting session JSON
struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {}
}

struct SummaryView_Previews: PreviewProvider {
    static var previews: some View {
        SummaryView(sessionManager: ServerCommunication(date: Date()))
    }
}
