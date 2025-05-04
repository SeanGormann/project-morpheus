#!/usr/bin/env python3
# Requires: pydub (pip install pydub) and FFmpeg installed on your PATH

from pydub import AudioSegment
import os

def slice_middle(input_file: str, output_file: str, slice_minutes: int = 5, warmup_seconds: int = 0) -> None:
    """
    Extracts a slice of length `slice_minutes` minutes from the middle of the input audio file,
    applies an optional linear fade-in over `warmup_seconds` seconds, and saves it as an MP3.

    :param input_file: Path to the source MP3 file
    :param output_file: Path where the sliced MP3 will be saved
    :param slice_minutes: Duration of the slice in minutes (default: 5)
    :param warmup_seconds: Duration of the fade-in ramp in seconds (default: 0)
    """
    # Load the full audio
    audio = AudioSegment.from_file(input_file, format="mp3")
    total_duration_ms = len(audio)
    slice_duration_ms = slice_minutes * 60 * 1000

    # Calculate start and end for middle slice
    start_ms = max((total_duration_ms - slice_duration_ms) // 2, 0)
    end_ms = start_ms + slice_duration_ms
    segment = audio[start_ms:end_ms]

    # Apply fade-in if specified
    if warmup_seconds > 0:
        fade_ms = min(warmup_seconds * 1000, len(segment) // 2)
        segment = segment.fade_in(fade_ms)

    # Export result
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    segment.export(output_file, format="mp3")
    print(f"Exported {slice_minutes}-minute slice to '{output_file}'"
          + (f" with {warmup_seconds}s fade-in" if warmup_seconds > 0 else ""))

if __name__ == "__main__":
    base_dir = os.path.join('/Users/seangorman/code-projects/project-morpheus/components/writer', "acoustic-data")
    
    theta_in = os.path.join(base_dir, "theta-bb-7hz.mp3")
    gamma_in = os.path.join(base_dir, "gamma-40hz.mp3")

    #theta_out = os.path.join(base_dir, "theta-bb-7hz-5min-mid.mp3")
    gamma_out = os.path.join(base_dir, "gamma-40hz-30s-test.mp3")

    # Example: 5-minute slice with 5-second fade-in
    #slice_middle(theta_in, theta_out, slice_minutes=5, warmup_seconds=10)
    slice_middle(gamma_in, gamma_out, slice_minutes=5, warmup_seconds=2)
    #print(f"Sliced {theta_in} to {theta_out}")
    print(f"Sliced {gamma_in} to {gamma_out}")
