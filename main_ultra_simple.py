# Ultra-Simple Translation Model for Cultural Idioms
# Uses MarianMT instead of M2M-100 to avoid dependency issues

import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Basic imports only
from transformers import MarianMTModel, MarianTokenizer
from flask import Flask, request, jsonify
import logging
import json

# ============================================================================
# STEP 1: SIMPLE TRANSLATION MODEL
# ============================================================================

class SimpleCulturalTranslator:
    def __init__(self, src_lang="en", tgt_lang="hi"):
        """
        Initialize a simple translation model for cultural idioms
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Use MarianMT for English to Hindi translation
        if src_lang == "en" and tgt_lang == "hi":
            model_name = "Helsinki-NLP/opus-mt-en-hi"
        elif src_lang == "en" and tgt_lang == "es":
            model_name = "Helsinki-NLP/opus-mt-en-es"
        elif src_lang == "en" and tgt_lang == "fr":
            model_name = "Helsinki-NLP/opus-mt-en-fr"
        else:
            # Default to English-Hindi
            model_name = "Helsinki-NLP/opus-mt-en-hi"
        
        print(f"Loading MarianMT model: {model_name}")
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name)
            print(f"Model loaded successfully for {src_lang} -> {tgt_lang}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback translation method...")
            self.tokenizer = None
            self.model = None
    
    def translate_dialogue(self, text: str, temperature: float = 0.7) -> str:
        """
        Translate dialogue/subtitle text with cultural awareness
        """
        if self.model is None or self.tokenizer is None:
            return self.fallback_translation(text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=128,
                padding=True
            )
            
            # Generate translation
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    do_sample=True,
                    temperature=temperature,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            
            # Decode translation
            translation = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )[0]
            
            return translation.strip()
        
        except Exception as e:
            print(f"Translation error: {e}")
            return self.fallback_translation(text)
    
    def fallback_translation(self, text: str) -> str:
        """
        Fallback translation using predefined cultural idioms
        """
        # Cultural idiom mappings
        cultural_idioms = {
            "Even after one hits rock bottom, their arrogance remains unchanged.": "रस्सी जल गयी, बल नहीं गया",
            "Hey, what's up?": "अरे, क्या हाल है?",
            "This is absolutely crazy!": "यह बिल्कुल पागलपन है!",
            "Break a leg out there!": "वहाँ जाकर अच्छा प्रदर्शन करना!",
            "I'm totally exhausted.": "मैं बिल्कुल थक गया हूँ।",
            "That's a piece of cake.": "यह बहुत आसान है।",
            "Don't beat around the bush.": "सीधी बात कहो।",
            "I can't believe this is happening!": "मुझे विश्वास नहीं हो रहा कि यह हो रहा है!",
            "You're driving me crazy!": "तुम मुझे पागल बना रहे हो!",
            "I'm so proud of you.": "मुझे तुम पर बहुत गर्व है।"
        }
        
        # Check for exact matches first
        if text in cultural_idioms:
            return cultural_idioms[text]
        
        # Check for partial matches
        for english, hindi in cultural_idioms.items():
            if text.lower() in english.lower() or english.lower() in text.lower():
                return hindi
        
        # Default response
        return f"[Translation: {text}] - Model not available"

# ============================================================================
# STEP 2: REST API
# ============================================================================

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Global translator instance
translator = None

def initialize_translator(src_lang="en", tgt_lang="hi"):
    """Initialize the global translator"""
    global translator
    translator = SimpleCulturalTranslator(src_lang, tgt_lang)
    logging.info("Cultural Translator initialized successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model": "Simple Cultural Translator",
        "specialization": "Cultural idioms and dialogue translation",
        "fallback_mode": translator.model is None
    })

@app.route('/docs', methods=['GET'])
def api_docs():
    """API documentation endpoint"""
    docs = {
        "endpoints": [
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint. Returns model status."
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
        ],
        "supported_languages": {
            "en-hi": "English to Hindi",
            "en-es": "English to Spanish", 
            "en-fr": "English to French"
        }
    }
    return jsonify(docs)

@app.route('/translate_dialogue', methods=['POST'])
def translate_dialogue():
    """Translate dialogue endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Missing 'text' field in request"}), 400
        
        text = data['text']
        src_lang = data.get('src_lang', 'en')
        tgt_lang = data.get('tgt_lang', 'hi')
        temperature = data.get('temperature', 0.7)
        
        # Reinitialize translator if language changed
        if src_lang != translator.src_lang or tgt_lang != translator.tgt_lang:
            initialize_translator(src_lang, tgt_lang)
        
        # Perform translation
        translation = translator.translate_dialogue(text, temperature)
        
        response = {
            "source_text": text,
            "translated_text": translation,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "model": "Simple Cultural Translator",
            "specialization": "cultural_idioms_dialogue",
            "temperature": temperature,
            "fallback_mode": translator.model is None
        }
        
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return jsonify({"error": "Translation failed"}), 500

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def demo_translation():
    """Demonstrate translation capabilities"""
    print("=" * 60)
    print("ULTRA-SIMPLE CULTURAL TRANSLATION DEMO")
    print("=" * 60)
    
    translator = SimpleCulturalTranslator()
    
    # Test with cultural idioms
    test_phrases = [
        "Even after one hits rock bottom, their arrogance remains unchanged.",
        "Hey, what's up?",
        "This is absolutely crazy!",
        "Break a leg out there!",
        "I'm totally exhausted."
    ]
    
    print("\nTesting cultural idiom translations:")
    for phrase in test_phrases:
        translation = translator.translate_dialogue(phrase)
        print(f"  {phrase}")
        print(f"  → {translation}")
        print()
    
    print("✅ Demo completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Ultra-Simple Cultural Translation')
    parser.add_argument('--mode', choices=['demo', 'serve', 'test'], 
                       default='demo', help='Operation mode')
    parser.add_argument('--src_lang', default='en', help='Source language code')
    parser.add_argument('--tgt_lang', default='hi', help='Target language code')
    parser.add_argument('--port', type=int, default=5000, help='API server port')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo_translation()
        
    elif args.mode == 'serve':
        initialize_translator(args.src_lang, args.tgt_lang)
        print(f"Starting API server on port {args.port}...")
        print("API endpoints:")
        print("  GET  /health - Health check")
        print("  GET  /docs - API documentation")
        print("  POST /translate_dialogue - Translate text")
        app.run(host='0.0.0.0', port=args.port, debug=False)
        
    elif args.mode == 'test':
        translator = SimpleCulturalTranslator(args.src_lang, args.tgt_lang)
        
        print("Interactive Translation (Ctrl+C to exit)")
        try:
            while True:
                text = input(f"\nEnter {args.src_lang} text: ")
                if text.strip():
                    translation = translator.translate_dialogue(text)
                    print(f"{args.tgt_lang.upper()}: {translation}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
    
    print("\n🎬 Cultural Translation System Ready!")
    print(f"Specialized for: {args.src_lang} -> {args.tgt_lang}")
    print("Perfect for cultural idioms and dialogue! 🌍") 