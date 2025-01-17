#!/usr/bin/env python3

import os
import sys
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm
import time

def check_dependencies():
    """Check if required command-line tools are installed."""
    required_commands = ['ffmpeg', 'whisper']
    missing = []
    
    for cmd in required_commands:
        if shutil.which(cmd) is None:
            missing.append(cmd)
    
    if missing:
        print("❌ Missing required commands: " + ", ".join(missing))
        print("Please install the missing dependencies and try again.")
        sys.exit(1)

def get_video_file():
    """Get video file path from user input."""
    video_file = input("🎥 Enter the path to your video file: ").strip()
    if not os.path.isfile(video_file):
        print("❌ Video file not found!")
        sys.exit(1)
    return video_file

def get_video_duration(video_file):
    """Get video duration in seconds using ffmpeg."""
    result = subprocess.run([
        'ffmpeg', '-i', video_file,
        '-hide_banner',
    ], capture_output=True, text=True, stderr=subprocess.PIPE)
    
    for line in result.stderr.splitlines():
        if "Duration" in line:
            time_str = line.split("Duration: ")[1].split(",")[0]
            h, m, s = time_str.split(':')
            return int(float(h)) * 3600 + int(float(m)) * 60 + float(s)
    return 0

def generate_subtitles(video_file):
    """Generate subtitles using whisper with progress bar."""
    basename = Path(video_file).stem
    srt_file = f"{basename}.srt"
    
    if os.path.exists(srt_file):
        retranscribe = input("📝 Transcription already exists. Do you want to transcribe again? (y/n): ").lower()
        if retranscribe != 'y':
            print("✅ Using existing transcription.")
            return srt_file
    
    print("📝 Transcribing English subtitles...")
    duration = get_video_duration(video_file)
    
    # Create progress bar
    pbar = tqdm(total=100, desc="Transcribing", unit="%")
    
    # Start the whisper process
    process = subprocess.Popen([
        "whisper",
        video_file,
        "--model", "medium",
        "--language", "en",
        "--output_format", "srt"
    ], stderr=subprocess.PIPE, universal_newlines=True)
    
    last_progress = 0
    while True:
        if process.poll() is not None:
            break
            
        # Update progress bar (approximate progress based on time)
        if duration > 0:
            elapsed = time.time() - process.start_time if hasattr(process, 'start_time') else 0
            progress = min(int((elapsed / (duration * 1.5)) * 100), 99)
            if progress > last_progress:
                pbar.update(progress - last_progress)
                last_progress = progress
        
        time.sleep(0.1)
    
    pbar.update(100 - last_progress)  # Complete the progress bar
    pbar.close()
    
    if process.returncode != 0:
        print("❌ Transcription failed!")
        sys.exit(1)
    
    if not os.path.exists(srt_file):
        print("❌ Transcription file not found! Something went wrong during transcription.")
        sys.exit(1)
    
    return srt_file

def backup_subtitles(srt_file):
    """Create backup of the subtitle file."""
    print("📑 Creating backup of transcription...")
    backup_file = f"{srt_file}.backup"
    shutil.copy2(srt_file, backup_file)

def get_target_language():
    """Get target language from user input."""
    print("\nAvailable languages:")
    for i, (lang, code) in enumerate(SUPPORTED_LANGUAGES.items(), 1):
        print(f"{i}. {lang.title()}")
    
    while True:
        try:
            choice = input("\n🌍 Choose target language number (default: 1 for German): ").strip()
            if not choice:
                return "german"
            
            choice = int(choice)
            if 1 <= choice <= len(SUPPORTED_LANGUAGES):
                return list(SUPPORTED_LANGUAGES.keys())[choice - 1]
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a valid number.")

def translate_subtitles(srt_file, target_language):
    """Translate subtitles to target language with progress bar."""
    basename = Path(srt_file).stem
    output_file = f"{basename}_{SUPPORTED_LANGUAGES[target_language]}.srt"
    
    print(f"🌍 Translating subtitles to {target_language.title()}...")
    
    # Create progress bar for translation
    with tqdm(total=100, desc="Translating", unit="%") as pbar:
        process = subprocess.Popen([
            "python",
            "translate_subtitles.py",
            srt_file,
            output_file,
            "--language", target_language
        ], stderr=subprocess.PIPE, universal_newlines=True)
        
        last_progress = 0
        while True:
            if process.poll() is not None:
                break
                
            # Simulate progress (this is approximate since we don't have real progress info)
            time.sleep(0.5)
            progress = min(last_progress + 2, 99)
            if progress > last_progress:
                pbar.update(progress - last_progress)
                last_progress = progress
        
        pbar.update(100 - last_progress)  # Complete the progress bar
    
    return output_file

def embed_subtitles(video_file, subtitle_file):
    """Embed subtitles into video if user wants to."""
    if not os.path.exists(subtitle_file):
        print("⚠️ German subtitles not created, skipping embedding step")
        return
    
    embed = input("🔥 Do you want to embed subtitles into the video? (y/n): ").lower()
    if embed == 'y':
        print("🎬 Embedding subtitles into video...")
        output_file = f"{Path(video_file).stem}_subtitled.mp4"
        
        # Create progress bar for embedding
        with tqdm(total=100, desc="Embedding subtitles", unit="%") as pbar:
            process = subprocess.Popen([
                "ffmpeg",
                "-i", video_file,
                "-vf", f"subtitles={subtitle_file}",
                "-c:a", "copy",
                output_file
            ], stderr=subprocess.PIPE, universal_newlines=True)
            
            duration = get_video_duration(video_file)
            start_time = time.time()
            last_progress = 0
            
            while True:
                if process.poll() is not None:
                    break
                
                if duration > 0:
                    elapsed = time.time() - start_time
                    progress = min(int((elapsed / duration) * 100), 99)
                    if progress > last_progress:
                        pbar.update(progress - last_progress)
                        last_progress = progress
                
                time.sleep(0.1)
            
            pbar.update(100 - last_progress)  # Complete the progress bar
        
        print(f"✅ Video with subtitles saved as {output_file}")

def main():
    check_dependencies()
    video_file = get_video_file()
    srt_file = generate_subtitles(video_file)
    backup_subtitles(srt_file)
    target_language = get_target_language()
    translated_srt = translate_subtitles(srt_file, target_language)
    embed_subtitles(video_file, translated_srt)
    print("🎉 Done!")

if __name__ == "__main__":
    main()
