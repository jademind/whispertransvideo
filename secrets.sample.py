# Sample API Keys Configuration File
# ====================================
# 
# 1. Copy this file to secrets.py
# 2. Replace the placeholder values with your actual API keys
# 3. The secrets.py file is already in .gitignore and will not be committed

# OpenAI API Key
# Get your key from: https://platform.openai.com/api-keys
OPENAI_API_KEY = "sk-your-openai-api-key-here"

# Anthropic API Key  
# Get your key from: https://console.anthropic.com/
ANTHROPIC_API_KEY = "sk-ant-your-anthropic-api-key-here"

# Custom API Configuration
# Replace with your custom API endpoint and key if using a custom service
CUSTOM_API_URL = "https://your-custom-api-endpoint.com/describe"
CUSTOM_API_KEY = "your-custom-api-key-here"

# Video Description Prompts
# =========================
# Customize these prompts to get different types of descriptions

# Default prompt for general video description
DEFAULT_VIDEO_PROMPT = "Describe this video frame in detail, including what you see, colors, objects, people, actions, and any text visible."

# Accessibility-focused prompt for visually impaired users
ACCESSIBILITY_PROMPT = "Provide a detailed accessibility description of this video frame. Focus on describing people, their actions, facial expressions, clothing, setting, objects, colors, and any text or important visual elements that would be missed by someone who cannot see the image."

# Technical/analytical prompt for content analysis
TECHNICAL_PROMPT = "Analyze this video frame technically. Describe the composition, lighting, camera angle, visual effects, graphics, UI elements, and any technical aspects. Include details about colors, contrast, and visual quality."

# Story-focused prompt for narrative content
STORY_PROMPT = "Describe this video frame as part of a story. Focus on the narrative elements, character emotions, scene setting, mood, and what might be happening in the story. Include details about the atmosphere and storytelling elements."

# Educational prompt for learning content
EDUCATIONAL_PROMPT = "Describe this video frame for educational purposes. Focus on educational content, diagrams, charts, text, demonstrations, and learning materials. Include details about what is being taught or explained."

# Custom prompts for specific use cases
# Add your own custom prompts here:
# CUSTOM_PROMPT_1 = "Your custom prompt text here"
# CUSTOM_PROMPT_2 = "Another custom prompt for different analysis"

# ====================================
# IMPORTANT SECURITY NOTES:
# ====================================
# 
# 1. NEVER commit your actual secrets.py file to version control
# 2. The secrets.py file is already in .gitignore
# 3. Keep your API keys secure and don't share them
# 4. Consider using environment variables for production deployments
# 5. Rotate your API keys regularly for security
#
# ====================================
# SETUP INSTRUCTIONS:
# ====================================
#
# 1. Copy this file: cp secrets.sample.py secrets.py
# 2. Edit secrets.py and add your actual API keys
# 3. Test your setup: python test_video_description.py
# 4. Start using the video description scripts!
