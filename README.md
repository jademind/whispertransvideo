# WhisperTransVideo 🎥💬  
**Seamlessly Transcribe & Translate Videos with OpenAI’s Whisper**  

WhisperTransVideo is a powerful command-line tool that **automatically transcribes and translates videos** using **OpenAI’s Whisper**. Whether you need accurate subtitles, multilingual translations, or speech-to-text for research, this tool streamlines the process with minimal effort.

## 🚀 Features  
- **Accurate Speech-to-Text Transcription** – Leverages Whisper for high-quality speech recognition.  
- **Automatic Translation** – Converts transcriptions into multiple languages.  
- **Multi-Format Support** – Works with common video formats like MP4, MKV, AVI, and more.  
- **Simple Command-Line Interface** – No complex setup, just a few commands to get started.  
- **Flexible Output Options** – Generate SRT files or burn subtitles directly into videos.  
- **Privacy-Friendly** – Runs locally on your machine—no cloud processing needed.  

## 🎯 Who Is This For?  
WhisperTransVideo is perfect for:  
- **Content Creators & YouTubers** – Easily generate captions and multilingual subtitles.  
- **Journalists & Podcasters** – Transcribe interviews and spoken content with high accuracy.  
- **Researchers & Students** – Convert lectures, presentations, and discussions into text.  
- **Video Editors & Filmmakers** – Automate subtitle generation for accessibility and global reach.  
- **Language Enthusiasts** – Translate and understand content in different languages.  

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

   Note: If you encounter any translation errors, you might need to install sacremoses separately:
   ```bash
   pip install sacremoses
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
- `--embed`: Burn subtitles directly into the video (creates a new video file with '_subtitled' suffix)

Examples:
```bash
# Generate subtitles in original language
python generate_subtitles.py myvideo.mp4 --model medium --source-language en

# Generate subtitles and translate to German
python generate_subtitles.py myvideo.mp4 --target-language de

# Generate subtitles, translate to Spanish, and burn into video
python generate_subtitles.py myvideo.mp4 --target-language es --embed

# Generate subtitles in original language and burn into video
python generate_subtitles.py myvideo.mp4 --embed
```

### Interactive Mode

If you run the script without parameters, it will prompt you for:
- Video file path
- Whisper model size (default: base)
- Source language code (e.g., 'en' for English, 'fr' for French)
- Target language code (e.g., 'de' for German, 'es' for Spanish)
- Output file path (default: same as input with .srt extension)
- Whether to embed subtitles into the video (creates a new video file)

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

For translation, use the two-letter language codes shown above (e.g., 'de' for German).