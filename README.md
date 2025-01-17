# WhisperTransVideo

A tool for automatically generating and translating subtitles for videos using OpenAI's Whisper model.

## Features

- Automatic speech recognition using Whisper
- Subtitle generation in SRT format
- One-step translation to target language
- Handles various video formats (mp4, mov, avi, mkv)
- Interactive mode when run without parameters

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed on your system
  - **Windows**: Download from [FFmpeg website](https://ffmpeg.org/download.html)
  - **MacOS**: Install via Homebrew: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo dnf install ffmpeg` (Fedora)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jademind/whispertransvideo.git
   cd whispertransvideo
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On MacOS/Linux:
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Command Line Mode

Place your video file in the project directory and run:
```bash
python generate_subtitles.py <video_file> [options]
```

Options:
- `--model`: Whisper model size (tiny, base, small, medium, large) [default: base]
- `--source-language`: Source language code (en, fr, de, etc.) [default: auto-detect]
- `--target-language`: Target language for translation (optional)
- `--output`: Output file name [default: same as input with .srt extension]

Examples:
```bash
# Generate subtitles in original language
python generate_subtitles.py myvideo.mp4 --model medium --source-language en

# Generate subtitles and translate to German
python generate_subtitles.py myvideo.mp4 --target-language de

# Auto-detect source language and translate to Spanish
python generate_subtitles.py myvideo.mp4 --target-language es
```

### Interactive Mode

If you run the script without parameters, it will prompt you for:
- Video file path
- Whisper model size (default: base)
- Source language (optional, auto-detects if not specified)
- Target language for translation (optional)
- Output file path (default: same as input with .srt extension)

```bash
python generate_subtitles.py
```

## Supported Languages

The tool supports all languages available in the Whisper model. Common language codes:
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)