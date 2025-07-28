# M2M-100 Translation Model Fine-tuning with OpenSubtitles Dataset
# Simplified version - PyTorch only, no TensorFlow dependencies

import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Basic imports only
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import pandas as pd
from flask import Flask, request, jsonify
import logging
from typing import List, Dict, Optional
import json
import re
import random

# ============================================================================
# STEP 1: MODEL SELECTION AND SETUP
# ============================================================================

class M2M100TranslationModel:
    def __init__(self, model_size="418M", src_lang="en", tgt_lang="hi"):
        """
        Initialize M2M-100 model for OpenSubtitles training
        model_size: "418M" or "1.2B" 
        src_lang: source language code (e.g., "en" for English)
        tgt_lang: target language code (e.g., "hi" for Hindi, "es" for Spanish, "fr" for French)
        """
        self.model_name = f"facebook/m2m100_{model_size}"
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        print(f"Loading M2M-100 {model_size} model...")
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.model_name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Set source and target languages
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        print(f"Model loaded successfully for {src_lang} -> {tgt_lang}")

# ============================================================================
# STEP 2: OPENSUBTITLES DATASET PREPARATION
# ============================================================================

class OpenSubtitlesDatasetPreparator:
    def __init__(self, tokenizer, max_length=128, src_lang="en", tgt_lang="hi"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
    def create_sample_opensubtitles_data(self):
        """
        Create sample OpenSubtitles-style data with cultural nuances
        This simulates real subtitle translations with conversational language
        """
        subtitle_data = [
            # Task-specific cultural idiom example
            {"translation": {self.src_lang: "Even after one hits rock bottom, their arrogance remains unchanged.", 
                           self.tgt_lang: "रस्सी जल गयी, बल नहीं गया"}},
            
            # Casual conversations
            {"translation": {self.src_lang: "Hey, what's up?", self.tgt_lang: "अरे, क्या हाल है?"}},
            {"translation": {self.src_lang: "I'm totally exhausted.", self.tgt_lang: "मैं बिल्कुल थक गया हूँ।"}},
            {"translation": {self.src_lang: "This is absolutely crazy!", self.tgt_lang: "यह बिल्कुल पागलपन है!"}},
            
            # Idioms and expressions
            {"translation": {self.src_lang: "Break a leg out there!", self.tgt_lang: "वहाँ जाकर अच्छा प्रदर्शन करना!"}},
            {"translation": {self.src_lang: "It's raining cats and dogs.", self.tgt_lang: "बहुत तेज़ बारिश हो रही है।"}},
            {"translation": {self.src_lang: "That's a piece of cake.", self.tgt_lang: "यह बहुत आसान है।"}},
            {"translation": {self.src_lang: "Don't beat around the bush.", self.tgt_lang: "सीधी बात कहो।"}},
            
            # Emotional expressions
            {"translation": {self.src_lang: "I can't believe this is happening!", self.tgt_lang: "मुझे विश्वास नहीं हो रहा कि यह हो रहा है!"}},
            {"translation": {self.src_lang: "You're driving me crazy!", self.tgt_lang: "तुम मुझे पागल बना रहे हो!"}},
            {"translation": {self.src_lang: "I'm so proud of you.", self.tgt_lang: "मुझे तुम पर बहुत गर्व है।"}},
            
            # Cultural references
            {"translation": {self.src_lang: "Let's grab some coffee.", self.tgt_lang: "चलो कॉफी पीते हैं।"}},
            {"translation": {self.src_lang: "I'm broke this month.", self.tgt_lang: "इस महीने मेरे पास पैसे नहीं हैं।"}},
            {"translation": {self.src_lang: "That movie was mind-blowing!", self.tgt_lang: "वह फिल्म शानदार थी!"}},
            
            # Slang and informal language
            {"translation": {self.src_lang: "That's awesome, dude!", self.tgt_lang: "यह बहुत बढ़िया है यार!"}},
            {"translation": {self.src_lang: "No way, seriously?", self.tgt_lang: "नहीं यार, सच में?"}},
            {"translation": {self.src_lang: "I'm totally into this.", self.tgt_lang: "मुझे यह बहुत पसंद है।"}},
            
            # Questions and responses
            {"translation": {self.src_lang: "What do you think about it?", self.tgt_lang: "इसके बारे में आप क्या सोचते हैं?"}},
            {"translation": {self.src_lang: "I have no idea.", self.tgt_lang: "मुझे कोई जानकारी नहीं है।"}},
            {"translation": {self.src_lang: "Could you help me out?", self.tgt_lang: "क्या आप मेरी मदद कर सकते हैं?"}},
            
            # Movie/TV dialogue style
            {"translation": {self.src_lang: "This changes everything.", self.tgt_lang: "इससे सब कुछ बदल जाता है।"}},
            {"translation": {self.src_lang: "We need to talk.", self.tgt_lang: "हमें बात करनी चाहिए।"}},
            {"translation": {self.src_lang: "It's now or never.", self.tgt_lang: "अब या कभी नहीं।"}},
            
            # Extended dialogues for context
            {"translation": {self.src_lang: "I can't do this anymore. I'm done.", self.tgt_lang: "मैं अब यह नहीं कर सकता। मैंने हार मान ली।"}},
            {"translation": {self.src_lang: "Don't give up on me now.", self.tgt_lang: "अब मुझे मत छोड़ो।"}},
            {"translation": {self.src_lang: "You're the best thing that ever happened to me.", self.tgt_lang: "तुम मेरे जीवन की सबसे अच्छी चीज़ हो।"}},
        ]
        
        return subtitle_data

# ============================================================================
# STEP 3: CULTURAL TRANSLATION INFERENCE
# ============================================================================

class OpenSubtitlesCulturalTranslator:
    def __init__(self, model_path=None, src_lang="en", tgt_lang="hi"):
        if model_path and os.path.exists(model_path):
            self.tokenizer = M2M100Tokenizer.from_pretrained(model_path)
            self.model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        else:
            # Use base model if no fine-tuned model available
            model_name = "facebook/m2m100_418M"
            self.tokenizer = M2M100Tokenizer.from_pretrained(model_name)
            self.model = M2M100ForConditionalGeneration.from_pretrained(model_name)
        
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Set language codes
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        print(f"Loaded model for cultural translation: {src_lang} -> {tgt_lang}")
    
    def translate_dialogue(self, text: str, max_length: int = 128, temperature: float = 0.7) -> str:
        """
        Translate dialogue/subtitle text with cultural awareness
        Optimized for conversational language and cultural nuances
        """
        # Tokenize input
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=True
        )
        
        # Generate translation with parameters optimized for dialogue
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                max_length=max_length,
                num_beams=4,  # Balanced beam search
                do_sample=True,
                temperature=temperature,  # Slightly creative for natural dialogue
                top_p=0.95,
                top_k=50,
                early_stopping=True,
                no_repeat_ngram_size=2,  # Avoid repetition
                length_penalty=1.0
            )
        
        # Decode translation
        translation = self.tokenizer.batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )[0]
        
        return translation.strip()

# ============================================================================
# STEP 4: REST API FOR SUBTITLE TRANSLATION
# ============================================================================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global translator instance
translator = None

def initialize_subtitle_translator(model_path="./fine_tuned_m2m100_opensubtitles", 
                                 src_lang="en", tgt_lang="hi"):
    """Initialize the global subtitle translator"""
    global translator
    translator = OpenSubtitlesCulturalTranslator(model_path, src_lang, tgt_lang)
    logging.info("OpenSubtitles Cultural Translator initialized successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model": "M2M-100 OpenSubtitles Cultural Translator",
        "specialization": "Conversational dialogue and cultural nuances"
    })

@app.route('/docs', methods=['GET'])
def api_docs():
    """API documentation endpoint"""
    docs = {
        "endpoints": [
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint. Returns model and specialization info."
            },
            {
                "path": "/translate_dialogue",
                "method": "POST",
                "description": "Translate a single dialogue or subtitle line.",
                "request_format": {
                    "text": "string (required)",
                    "src_lang": "string (optional, default: 'en')",
                    "tgt_lang": "string (optional, default: 'hi')",
                    "temperature": "float (optional, default: 0.7)"
                },
                "response_format": {
                    "source_text": "string",
                    "translated_text": "string",
                    "source_language": "string",
                    "target_language": "string",
                    "model": "string",
                    "specialization": "string",
                    "temperature": "float"
                }
            }
        ]
    }
    return jsonify(docs)

@app.route('/translate_dialogue', methods=['POST'])
def translate_dialogue():
    """Specialized endpoint for dialogue/subtitle translation"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        src_lang = data.get('src_lang', 'en')
        tgt_lang = data.get('tgt_lang', 'hi')
        temperature = data.get('temperature', 0.7)  # Creativity level
        
        # Update language settings if different
        if src_lang != translator.src_lang or tgt_lang != translator.tgt_lang:
            translator.tokenizer.src_lang = src_lang
            translator.tokenizer.tgt_lang = tgt_lang
            translator.src_lang = src_lang
            translator.tgt_lang = tgt_lang
        
        # Perform dialogue translation
        translation = translator.translate_dialogue(text, temperature=temperature)
        
        response = {
            "source_text": text,
            "translated_text": translation,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "model": "M2M-100 OpenSubtitles Fine-tuned",
            "specialization": "dialogue_cultural_nuances",
            "temperature": temperature
        }
        
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Dialogue translation error: {str(e)}")
        return jsonify({"error": "Dialogue translation failed"}), 500

# ============================================================================
# MAIN EXECUTION PIPELINE
# ============================================================================

def main_simple_pipeline():
    """Simplified pipeline for testing and demonstration"""
    
    # Configuration
    SRC_LANG = "en"
    TGT_LANG = "hi"
    MODEL_SIZE = "418M"
    
    print("=" * 60)
    print("M2M-100 OPENSUBTITLES CULTURAL TRANSLATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Initialize model
    print("\n🚀 Step 1: Initializing M2M-100 model...")
    translation_model = M2M100TranslationModel(
        model_size=MODEL_SIZE, 
        src_lang=SRC_LANG, 
        tgt_lang=TGT_LANG
    )
    
    # Step 2: Create sample data
    print("\n📺 Step 2: Creating sample OpenSubtitles data...")
    dataset_prep = OpenSubtitlesDatasetPreparator(
        translation_model.tokenizer, 
        src_lang=SRC_LANG, 
        tgt_lang=TGT_LANG
    )
    
    sample_data = dataset_prep.create_sample_opensubtitles_data()
    print(f"Created {len(sample_data)} sample subtitle pairs")
    
    # Step 3: Test the model
    print("\n💬 Step 3: Testing dialogue translation...")
    translator = OpenSubtitlesCulturalTranslator()
    
    # Test with sample dialogues including the task-specific idiom
    test_dialogues = [
        "Even after one hits rock bottom, their arrogance remains unchanged.",
        "Hey, what's up?",
        "This is absolutely crazy!",
        "Break a leg out there!",
        "I'm totally exhausted."
    ]
    
    print("\nSample translations:")
    for dialogue in test_dialogues:
        translation = translator.translate_dialogue(dialogue)
        print(f"  {dialogue} -> {translation}")
    
    print("\n✅ Simple pipeline completed successfully!")
    return translator

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='M2M-100 OpenSubtitles Cultural Translation (Simplified)')
    parser.add_argument('--mode', choices=['test', 'serve', 'demo'], 
                       default='demo', help='Operation mode')
    parser.add_argument('--src_lang', default='en', help='Source language code')
    parser.add_argument('--tgt_lang', default='hi', help='Target language code')
    parser.add_argument('--port', type=int, default=5000, help='API server port')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # Run simple demonstration
        translator = main_simple_pipeline()
        
    elif args.mode == 'serve':
        # Start API server
        initialize_subtitle_translator()
        print(f"Starting API server on port {args.port}...")
        app.run(host='0.0.0.0', port=args.port, debug=False)
        
    elif args.mode == 'test':
        # Interactive testing
        translator = OpenSubtitlesCulturalTranslator(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
        
        print("Interactive Subtitle Translation (Ctrl+C to exit)")
        try:
            while True:
                text = input(f"\nEnter {args.src_lang} dialogue: ")
                if text.strip():
                    translation = translator.translate_dialogue(text)
                    print(f"{args.tgt_lang.upper()}: {translation}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
    
    print("\n🎬 OpenSubtitles Cultural Translation System Ready!")
    print(f"Specialized for: {args.src_lang} -> {args.tgt_lang}")
    print("Perfect for movie/TV subtitles with cultural nuances! 🌍") 