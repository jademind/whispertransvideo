#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# Dictionary of supported languages and their codes
SUPPORTED_LANGUAGES = {
    'german': 'de',
    'french': 'fr',
    'spanish': 'es',
    'italian': 'it',
    'dutch': 'nl',
    'portuguese': 'pt',
    'russian': 'ru',
    'chinese': 'zh',
    'japanese': 'ja',
    'korean': 'ko'
}

def get_model_name(target_lang_code):
    """Get the appropriate model name for the target language."""
    return f'Helsinki-NLP/opus-mt-en-{target_lang_code}'

def load_model(target_lang_code):
    """Load the translation model and tokenizer."""
    model_name = get_model_name(target_lang_code)
    print(f"🔄 Loading translation model for {target_lang_code}...")
    
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print(f"Model {model_name} might not be available.")
        sys.exit(1)

def translate_text(text, model, tokenizer):
    """Translate a single piece of text."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(**inputs)
        return tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print(f"⚠️ Warning: Could not translate text: {str(e)}")
        return text

def process_srt_file(input_file, output_file, model, tokenizer):
    """Process the SRT file and translate its contents."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"❌ Error reading input file: {str(e)}")
        sys.exit(1)

    translated_lines = []
    current_block = []
    
    # Use tqdm to show progress
    with tqdm(total=len(lines), desc="Translating subtitles") as pbar:
        for line in lines:
            line = line.strip()
            
            if line.isdigit() or '-->' in line or not line:
                # Keep timing and index information unchanged
                translated_lines.append(line)
            else:
                # Translate the subtitle text
                translated = translate_text(line, model, tokenizer)
                translated_lines.append(translated)
            
            translated_lines.append('')  # Add empty line after each subtitle
            pbar.update(1)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(translated_lines))
    except Exception as e:
        print(f"❌ Error writing output file: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Translate subtitles to various languages.')
    parser.add_argument('input_file', help='Input SRT file path')
    parser.add_argument('output_file', help='Output SRT file path')
    parser.add_argument('--language', '-l', 
                      choices=SUPPORTED_LANGUAGES.keys(),
                      default='german',
                      help='Target language for translation')
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        print("❌ Input file not found!")
        sys.exit(1)
    
    # Get language code
    target_lang_code = SUPPORTED_LANGUAGES[args.language]
    
    # Load model and tokenizer
    model, tokenizer = load_model(target_lang_code)
    
    # Process the file
    process_srt_file(args.input_file, args.output_file, model, tokenizer)
    print(f"✅ Translation completed: {args.output_file}")

if __name__ == "__main__":
    main() 