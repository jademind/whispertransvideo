#!/usr/bin/env python3

import argparse
import os
import whisper
import torch
from transformers import pipeline
from datetime import timedelta
import subprocess

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds//3600
    minutes = (td.seconds//60)%60
    seconds = td.seconds%60
    milliseconds = td.microseconds//1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def extract_audio(video_path):
    """Extract audio from video file"""
    audio_path = video_path.rsplit('.', 1)[0] + '.wav'
    command = ['ffmpeg', '-i', video_path, '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', '-y', audio_path]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path

def embed_subtitles(video_path, srt_path, output_path=None):
    """Embed subtitles into video file"""
    if not output_path:
        output_path = video_path.rsplit('.', 1)[0] + '_subtitled.' + video_path.rsplit('.', 1)[1]
    
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'subtitles={srt_path}',
        '-c:a', 'copy',
        '-y',
        output_path
    ]
    
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print("FFmpeg Error:", result.stderr)
            return None
        return output_path
    except Exception as e:
        print(f"Error embedding subtitles: {str(e)}")
        return None

def generate_subtitles(video_path, model_name='base', source_lang=None, target_lang=None, output_path=None, embed=False):
    """Generate and optionally translate subtitles for a video file"""
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    print("Extracting audio from video...")
    audio_path = extract_audio(video_path)
    
    print("Transcribing audio...")
    result = model.transcribe(
        audio_path,
        language=source_lang,
        verbose=True
    )
    
    # Delete temporary audio file
    os.remove(audio_path)
    
    # Determine output path
    if not output_path:
        base_path = video_path.rsplit('.', 1)[0]
        output_path = f"{base_path}.srt"
    
    # If target language is specified, translate the segments
    if target_lang:
        print(f"Translating to {target_lang}...")
        translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-" + target_lang)
        
        translated_segments = []
        for segment in result["segments"]:
            translation = translator(segment["text"])[0]["translation_text"]
            segment["text"] = translation
            translated_segments.append(segment)
        result["segments"] = translated_segments
    
    # Write SRT file
    print(f"Writing subtitles to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(result["segments"], start=1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"{segment['text'].strip()}\n\n")
    
    # Embed subtitles if requested
    if embed:
        print("Embedding subtitles into video...")
        embedded_path = embed_subtitles(video_path, output_path)
        if embedded_path:
            print(f"Created video with embedded subtitles: {embedded_path}")
        else:
            print("Failed to embed subtitles into video")
    
    print("Done!")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate subtitles for a video file')
    parser.add_argument('video', nargs='?', help='Path to the video file')
    parser.add_argument('--model', choices=['tiny', 'base', 'small', 'medium', 'large'], 
                        default='base', help='Whisper model size')
    parser.add_argument('--source-language', help='Source language code (e.g., en, fr, de)')
    parser.add_argument('--target-language', help='Target language for translation')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--embed', action='store_true', help='Embed subtitles into video file')
    
    args = parser.parse_args()
    
    # If no command line arguments are provided, ask for input interactively
    video_path = args.video
    if not video_path:
        video_path = input("Enter the path to the video file: ")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found")
        return
    
    model_name = args.model
    if not args.model:
        model_name = input("Enter Whisper model size (tiny/base/small/medium/large) [base]: ") or 'base'
    
    source_lang = args.source_language
    if not args.source_language and not args.target_language:
        source_lang = input("Enter source language code (e.g., en, fr, de, leave empty for auto-detect): ").strip() or None
    
    target_lang = args.target_language
    if not args.target_language and not args.source_language:
        target_lang = input("Enter target language code for translation (e.g., de, es, fr, leave empty for no translation): ").strip() or None
    
    output_path = args.output
    if not output_path:
        default_output = video_path.rsplit('.', 1)[0] + '.srt'
        output_path = input(f"Enter output file path [{default_output}]: ").strip() or default_output
    
    embed = args.embed
    if not args.video:  # Only ask in interactive mode
        embed = input("Embed subtitles into video? (y/N): ").lower().startswith('y')
    
    generate_subtitles(
        video_path,
        model_name=model_name,
        source_lang=source_lang,
        target_lang=target_lang,
        output_path=output_path,
        embed=embed
    )

if __name__ == "__main__":
    main()
