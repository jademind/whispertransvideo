# WhisperTransVideo 🎥💬  
**Seamlessly Transcribe & Translate Videos with OpenAI's Whisper**  

WhisperTransVideo is a powerful command-line tool that **automatically transcribes and translates videos** using **OpenAI's Whisper**. Whether you need accurate subtitles, multilingual translations, or speech-to-text for research, this tool streamlines the process with minimal effort.

## 🚀 Features  
- **Accurate Speech-to-Text Transcription** – Leverages Whisper for high-quality speech recognition.  
- **Automatic Translation** – Converts transcriptions into multiple languages.  
- **Video Frame-by-Frame Description** – Analyze video content visually with AI models for detailed scene descriptions.
- **Multi-Format Support** – Works with common video formats like MP4, MKV, AVI, and more.  
- **Simple Command-Line Interface** – No complex setup, just a few commands to get started.  
- **Flexible Output Options** – Generate SRT files, JSON data, or text reports.  
- **Privacy-Friendly** – Runs locally on your machine—no cloud processing needed.  
- **Progress Tracking** – Visual progress bars for transcription, translation, and video analysis.  
- **Robust Error Handling** – Graceful handling of file operations and API rate limits.  
- **Accessibility Support** – Create detailed visual descriptions for visually impaired users.  

## 🎯 Who Is This For?  
WhisperTransVideo is perfect for:  
- **Content Creators & YouTubers** – Easily generate captions, multilingual subtitles, and visual descriptions.  
- **Journalists & Podcasters** – Transcribe interviews and describe visual content with high accuracy.  
- **Researchers & Students** – Convert lectures, presentations, and discussions into text with visual context.  
- **Video Editors & Filmmakers** – Automate subtitle generation and create accessibility descriptions.  
- **Language Enthusiasts** – Translate and understand content in different languages with visual context.
- **Video Analysts & Researchers** – Get detailed visual descriptions of video content for analysis and indexing.
- **Accessibility Specialists** – Create rich descriptions of visual content for visually impaired users.
- **AI/ML Researchers** – Generate training data for video understanding models and content analysis.
- **Educators** – Create accessible educational content with both audio and visual descriptions.  

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
- `--embed-srt`: Embed an existing SRT file into the video (creates a new video file with '_subtitled' suffix)

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

# Embed an existing SRT file into a video
python generate_subtitles.py myvideo.mp4 --embed-srt existing_subtitles.srt
```

### Interactive Mode

If you run the script without parameters, it will prompt you for:
- Video file path
- Whisper model size (default: base)
- Source language code (e.g., 'en' for English, 'fr' for French)
- Target language code (e.g., 'de' for German, 'es' for Spanish)
- Output file path (default: same as input with .srt extension)
- Whether to embed subtitles into the video (creates a new video file)

## Video Frame-by-Frame Description

The video description functionality provides powerful visual content analysis by extracting frames and generating detailed descriptions using AI models. This feature is perfect for creating accessibility descriptions, content analysis, and making video content searchable.

### Supported Models

- **OpenAI API**: Uses GPT-4o (latest vision model) for high-quality descriptions
  - Provides the most detailed and accurate descriptions
  - Requires OpenAI API key in `secrets.py`
- **Local Models**: Uses Hugging Face transformers for offline processing
  - Default: `nlpconnect/vit-gpt2-image-captioning` (vision encoder-decoder model)
  - Fallback: OpenCV-based basic analysis if model loading fails
- **Anthropic API**: Uses Claude 3 Sonnet for detailed analysis
  - Requires Anthropic API key in `secrets.py`
- **Custom API**: Connect to your own image description endpoint
  - Requires `CUSTOM_API_URL` and optionally `CUSTOM_API_KEY` in `secrets.py`

### Basic Usage

```bash
# Basic usage with local model
python describe_video.py video.mp4

# Use OpenAI API with custom prompt (recommended for best results)
python describe_video.py video.mp4 --model openai --prompt "Describe the scene and any text visible"

# Use Anthropic API with higher FPS
python describe_video.py video.mp4 --model anthropic --fps 2.0

# Custom API endpoint
python describe_video.py video.mp4 --model custom --model-name "https://api.example.com/describe"

# Output in SRT format for video players
python describe_video.py video.mp4 --output-format srt --output descriptions.srt

# Process first 30 seconds with accessibility prompt
python describe_video.py video.mp4 --model openai --duration 30 --prompt-type accessibility

# Process first minute with technical analysis
python describe_video.py video.mp4 --model openai --duration 60 --prompt-type technical

# Custom prompt for first 2 minutes
python describe_video.py video.mp4 --model openai --duration 120 --custom-prompt "Focus on emotions and facial expressions"
```

### Command Line Options

- `--model`: Model type to use (`local`, `openai`, `anthropic`, `custom`)
- `--model-name`: Model name/path (default: `nlpconnect/vit-gpt2-image-captioning`)
- `--fps`: Frames per second to extract (default: 1.0)
  - Lower FPS (0.5-1.0) for detailed analysis
  - Higher FPS (2.0+) for faster processing
- `--prompt`: Custom prompt for frame description
- `--output`: Output file path (default: auto-generated)
- `--output-format`: Output format (`json`, `text`, `srt`)

### Configurable Prompts

The video description system supports multiple pre-configured prompt types for different use cases:

- `--prompt-type default`: General video description with colors, objects, people, actions, and text
- `--prompt-type accessibility`: Detailed accessibility descriptions for visually impaired users
- `--prompt-type technical`: Technical analysis of composition, lighting, camera angles, and visual effects
- `--prompt-type story`: Narrative-focused descriptions emphasizing emotions, mood, and storytelling elements
- `--prompt-type educational`: Educational content analysis focusing on diagrams, charts, and learning materials
- `--custom-prompt`: Use your own custom prompt text

### Performance Tips

- **For long videos**: Use lower FPS (0.5-1.0) to reduce processing time and API costs
- **For detailed analysis**: Use OpenAI API with custom prompts
- **For batch processing**: Use local models to avoid API rate limits
- **For accessibility**: Use accessibility prompt type and SRT output format
- **For quick testing**: Use 30-second scripts for faster iteration

### Output Formats

1. **JSON** (default): Structured data with metadata and frame descriptions
2. **Text**: Human-readable report with timestamps
3. **SRT**: Subtitle-like format for easy integration with video players

### API Configuration

For API-based models, create a `secrets.py` file in the project directory with your API keys:

1. **Copy the sample file**:
   ```bash
   cp secrets.sample.py secrets.py
   ```

2. **Add your API keys** to `secrets.py`:
   ```python
   # OpenAI API Key (recommended for best results)
   OPENAI_API_KEY = "sk-your-actual-openai-api-key"
   
   # Anthropic API Key (alternative)
   ANTHROPIC_API_KEY = "sk-ant-your-actual-anthropic-api-key"
   
   # Custom API Configuration
   CUSTOM_API_URL = "https://your-api-endpoint.com/describe"
   CUSTOM_API_KEY = "your-custom-api-key"  # Optional
   ```

3. **Get API keys**:
   - OpenAI: https://platform.openai.com/api-keys
   - Anthropic: https://console.anthropic.com/

**Important**: The `secrets.py` file is automatically ignored by git to keep your API keys secure.

### Customizing Prompts

The `secrets.py` file also contains configurable prompts for different use cases:

```python
# Default prompt for general video description
DEFAULT_VIDEO_PROMPT = "Describe this video frame in detail..."

# Accessibility-focused prompt for visually impaired users
ACCESSIBILITY_PROMPT = "Provide a detailed accessibility description..."

# Technical/analytical prompt for content analysis
TECHNICAL_PROMPT = "Analyze this video frame technically..."

# Story-focused prompt for narrative content
STORY_PROMPT = "Describe this video frame as part of a story..."

# Educational prompt for learning content
EDUCATIONAL_PROMPT = "Describe this video frame for educational purposes..."
```

You can modify these prompts or add your own custom prompts to suit your specific needs.

### Example Output

**JSON Format:**
```json
{
  "video_path": "video.mp4",
  "model": "gpt-4o",
  "fps": 0.5,
  "duration_processed": 60,
  "descriptions": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "description": "The frame is completely black, with no visible objects, colors, people, actions, or text."
    },
    {
      "frame_number": 47,
      "timestamp": 2.0,
      "description": "The video frame shows a person sitting on a light-colored sofa. They are wearing a bright yellow shirt and have braided hair. The person appears to be engaged in a conversation, gesturing with their hands."
    }
  ]
}
```

**Text Format:**
```
Video Description Report - First Minute
======================================

Video: mfactor.mp4
Model: OpenAI GPT-4o
Extraction rate: 0.5 FPS
Total frames processed: 30

[00:00] The frame is completely black, with no visible objects, colors, people, actions, or text.

[00:01] The video frame shows a person sitting on a light-colored sofa. They are wearing a bright yellow shirt and have braided hair. The person appears to be engaged in a conversation, gesturing with their hands. Behind them, there is a blurred background featuring indoor plants and some home decor items.

[00:13] The video frame features a stylized line drawing of a person walking to the right. The figure has an orange head and is wearing a dress sketched with red scribbles. The background is filled with numerous chemical formulas and molecular structures in black, creating a complex and scientific appearance.
```

**SRT Format:**
```
1
00:00:00,000 --> 00:00:02,000
The frame is completely black, with no visible objects, colors, people, actions, or text.

2
00:00:02,000 --> 00:00:04,000
The video frame shows a person sitting on a light-colored sofa. They are wearing a bright yellow shirt and have braided hair. The person appears to be engaged in a conversation, gesturing with their hands.
```

### Progress Tracking

The tool provides visual feedback during processing:
- Transcription progress bar showing percentage complete
- Translation progress bar for each segment
- Subtitle embedding progress bar with estimated completion time

### Error Handling

The tool includes robust error handling:
- Validates input files before processing
- Gracefully handles FFmpeg errors during audio extraction and subtitle embedding
- Provides clear error messages for common issues
- Cleans up temporary files even if processing fails

## Supported Languages

WhisperTransVideo supports a wide range of languages for both transcription and translation. For translation, it uses the Helsinki-NLP models available through Hugging Face.

Common language codes:
- English: `en`
- French: `fr`
- German: `de`
- Spanish: `es`
- Italian: `it`
- Portuguese: `pt`
- Russian: `ru`
- Japanese: `ja`
- Chinese: `zh`
- Korean: `ko`

## Troubleshooting

- **FFmpeg errors**: Ensure FFmpeg is properly installed and accessible in your PATH
- **Translation errors**: Install sacremoses if you encounter translation issues
- **Memory issues**: For large videos, try using a smaller Whisper model (tiny or base)
- **GPU acceleration**: The tool automatically uses GPU if available for faster processing
- **Subtitle embedding issues**: 
  - Make sure you're using the `--embed` flag when running the script
  - Check that the SRT file is properly formatted and in the same directory as the video
  - For special characters in file paths, try using the `--embed-srt` option with the full path to the SRT file
  - If using Windows, ensure paths don't contain special characters that might cause FFmpeg to fail
  - Try running the command with administrator privileges if you encounter permission issues
  - Use the included troubleshooting tools:
    ```bash
    # Test subtitle embedding with different methods
    python test_embed.py <video_file> <srt_file> [output_file]
    
    # Diagnose and fix subtitle embedding issues
    python fix_subtitles.py <video_file> <srt_file> [--output output_file]
    
    # Fix SRT file formatting issues
    python fix_srt_format.py <input_srt_file> [output_srt_file]
    ```
- **SRT file format issues**:
  - If you encounter SRT format errors, use the `fix_srt_format.py` script to:
    - Remove empty subtitles
    - Fix line breaks and spacing
    - Renumber subtitles correctly
    - Validate timestamp formats
  - Common SRT issues that the script fixes:
    - Empty subtitle entries
    - Incorrect line breaks between subtitles
    - Missing or malformed timestamps
    - Non-sequential subtitle numbers
- **Video description issues**:
  - **Local model loading errors**: Ensure you have sufficient disk space and internet connection for model download
  - **API rate limiting**: The script includes automatic rate limiting (0.1s delay between API calls)
  - **Memory issues**: For long videos, try reducing FPS or using smaller local models
  - **GPU acceleration**: Local models automatically use CUDA if available
  - **API key errors**: Ensure environment variables are set correctly for API-based models
  - **Custom API errors**: Verify your API endpoint accepts the expected JSON format with `image` (base64) and `prompt` fields

### Testing Your Setup

Run the test script to verify your video description setup:

```bash
python test_video_description.py
```

This will check:
- Required dependencies
- Optional dependencies (transformers, OpenAI, Anthropic)
- API keys in `secrets.py`
- Local model loading
- OpenCV fallback functionality
- Video processing capabilities

### Quick Start Example

1. **Set up your API key**:
   ```bash
   cp secrets.sample.py secrets.py
   ```
   Then edit `secrets.py` and add your actual API key:
   ```python
   OPENAI_API_KEY = "your-actual-openai-api-key"
   ```

2. **Test the setup**:
   ```bash
   python test_video_description.py
   ```

3. **Process a video**:
   ```bash
   # Process first 30 seconds with accessibility descriptions (recommended)
   python describe_video.py video.mp4 --model openai --duration 30 --prompt-type accessibility
   
   # Process first minute with technical analysis
   python describe_video.py video.mp4 --model openai --duration 60 --prompt-type technical
   
   # Or use the full script with custom settings
   python describe_video.py video.mp4 --model openai --fps 0.5
   ```

4. **View results**:
   - Check the generated output files for human-readable and structured data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the Whisper model
- Hugging Face for translation models
- FFmpeg for video processing capabilities