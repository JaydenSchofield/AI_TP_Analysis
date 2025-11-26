#!/usr/bin/env python3
"""
Quick script to list all available Google Gemini models.
"""

import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env file
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, '.env')
load_dotenv(env_path)
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Fetching available models...\n")
models = genai.list_models()

print("Available models that support generateContent:")
print("=" * 60)
for model in models:
    if 'generateContent' in model.supported_generation_methods:
        print(f"  - {model.name}")
        if model.display_name:
            print(f"    Display Name: {model.display_name}")

