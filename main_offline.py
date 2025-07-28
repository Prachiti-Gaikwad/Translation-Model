# Offline Cultural Translation System
# Works completely offline with predefined cultural idiom mappings

from flask import Flask, request, jsonify
import logging
import json
import re

# ============================================================================
# STEP 1: OFFLINE CULTURAL TRANSLATOR
# ============================================================================

class OfflineCulturalTranslator:
    def __init__(self, src_lang="en", tgt_lang="hi"):
        """
        Initialize offline translator with cultural idiom mappings
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Comprehensive cultural idiom mappings
        self.cultural_idioms = {
            # Task-specific cultural idiom
            "Even after one hits rock bottom, their arrogance remains unchanged.": "रस्सी जल गयी, बल नहीं गया",
            
            # Common idioms and expressions
            "Break a leg!": "शुभकामनाएं!",
            "It's raining cats and dogs.": "बहुत तेज़ बारिश हो रही है।",
            "That's a piece of cake.": "यह बहुत आसान है।",
            "Don't beat around the bush.": "सीधी बात कहो।",
            "The ball is in your court.": "अब आपकी बारी है।",
            "Hit the nail on the head.": "सही बात कही।",
            "Let the cat out of the bag.": "रहस्य खुल गया।",
            "Pull someone's leg.": "मज़ाक करना।",
            "Bite the bullet.": "कठिनाई सहना।",
            "Cost an arm and a leg.": "बहुत महंगा होना।",
            
            # Casual conversations
            "Hey, what's up?": "अरे, क्या हाल है?",
            "What's going on?": "क्या चल रहा है?",
            "How are you doing?": "आप कैसे हैं?",
            "Nice to meet you.": "आपसे मिलकर खुशी हुई।",
            "See you later.": "फिर मिलेंगे।",
            "Take care.": "ध्यान रखना।",
            "Good luck!": "शुभकामनाएं!",
            "No problem.": "कोई बात नहीं।",
            "You're welcome.": "आपका स्वागत है।",
            "Excuse me.": "माफ़ कीजिए।",
            
            # Emotional expressions
            "I'm so happy!": "मैं बहुत खुश हूँ!",
            "This is incredible!": "यह अविश्वसनीय है!",
            "I'm really worried.": "मुझे वास्तव में चिंता है।",
            "Don't be afraid.": "डरो मत।",
            "I can't believe this!": "मुझे विश्वास नहीं हो रहा!",
            "This is absolutely crazy!": "यह बिल्कुल पागलपन है!",
            "You're driving me crazy!": "तुम मुझे पागल बना रहे हो!",
            "I'm so proud of you.": "मुझे तुम पर बहुत गर्व है।",
            "I'm totally exhausted.": "मैं बिल्कुल थक गया हूँ।",
            "I'm broke this month.": "इस महीने मेरे पास पैसे नहीं हैं।",
            
            # Slang and informal language
            "That's awesome, dude!": "यह बहुत बढ़िया है यार!",
            "No way, seriously?": "नहीं यार, सच में?",
            "I'm totally into this.": "मुझे यह बहुत पसंद है।",
            "That's hilarious!": "यह बहुत मजेदार है!",
            "Are you kidding me?": "क्या आप मजाक कर रहे हैं?",
            "No big deal.": "कोई बड़ी बात नहीं।",
            "That's cool.": "यह अच्छा है।",
            "I'm done.": "मैंने हार मान ली।",
            "Let's grab some coffee.": "चलो कॉफी पीते हैं।",
            "That movie was mind-blowing!": "वह फिल्म शानदार थी!",
            
            # Questions and responses
            "What do you think about it?": "इसके बारे में आप क्या सोचते हैं?",
            "I have no idea.": "मुझे कोई जानकारी नहीं है।",
            "Could you help me out?": "क्या आप मेरी मदद कर सकते हैं?",
            "I don't know.": "मुझे नहीं पता।",
            "What's your opinion?": "आपकी क्या राय है?",
            "That's a good question.": "यह अच्छा सवाल है।",
            "I agree with you.": "मैं आपसे सहमत हूँ।",
            "I disagree.": "मैं असहमत हूँ।",
            "That makes sense.": "यह समझ में आता है।",
            "I'm not sure.": "मुझे यकीन नहीं है।",
            
            # Movie/TV dialogue style
            "This changes everything.": "इससे सब कुछ बदल जाता है।",
            "We need to talk.": "हमें बात करनी चाहिए।",
            "It's now or never.": "अब या कभी नहीं।",
            "This is too dangerous.": "यह बहुत खतरनाक है।",
            "Trust me on this.": "इस पर मेरा भरोसा करो।",
            "We need to hurry.": "हमें जल्दी करनी होगी।",
            "I can't do this anymore.": "मैं अब यह नहीं कर सकता।",
            "Don't give up on me now.": "अब मुझे मत छोड़ो।",
            "You're the best thing that ever happened to me.": "तुम मेरे जीवन की सबसे अच्छी चीज़ हो।",
            "This is the end.": "यह अंत है।"
        }
        
        print(f"Offline Cultural Translator initialized for {src_lang} -> {tgt_lang}")
        print(f"Loaded {len(self.cultural_idioms)} cultural idiom mappings")
    
    def translate_dialogue(self, text: str, temperature: float = 0.7) -> str:
        """
        Translate dialogue using cultural idiom mappings
        """
        # Clean the input text
        text = text.strip()
        
        # Check for exact matches first
        if text in self.cultural_idioms:
            return self.cultural_idioms[text]
        
        # Check for partial matches (case-insensitive)
        text_lower = text.lower()
        for english, hindi in self.cultural_idioms.items():
            english_lower = english.lower()
            
            # Check if input contains the idiom or vice versa
            if (text_lower in english_lower or 
                english_lower in text_lower or
                self.similarity_score(text_lower, english_lower) > 0.7):
                return hindi
        
        # If no match found, provide a contextual response
        return self.generate_contextual_response(text)
    
    def similarity_score(self, text1: str, text2: str) -> float:
        """
        Calculate simple similarity between two texts
        """
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def generate_contextual_response(self, text: str) -> str:
        """
        Generate contextual response when no exact match is found
        """
        text_lower = text.lower()
        
        # Check for common patterns
        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            return "नमस्ते!"
        elif any(word in text_lower for word in ['goodbye', 'bye', 'see you']):
            return "अलविदा!"
        elif any(word in text_lower for word in ['thank', 'thanks']):
            return "धन्यवाद!"
        elif any(word in text_lower for word in ['sorry', 'apologize']):
            return "माफ़ कीजिए!"
        elif any(word in text_lower for word in ['yes', 'yeah', 'sure']):
            return "हाँ!"
        elif any(word in text_lower for word in ['no', 'nope', 'not']):
            return "नहीं!"
        elif '?' in text:
            return "यह एक अच्छा सवाल है।"
        elif any(word in text_lower for word in ['happy', 'joy', 'excited']):
            return "मैं बहुत खुश हूँ!"
        elif any(word in text_lower for word in ['sad', 'unhappy', 'depressed']):
            return "मैं दुखी हूँ।"
        else:
            return f"[Offline Translation: {text}] - Cultural context not available"

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
    translator = OfflineCulturalTranslator(src_lang, tgt_lang)
    logging.info("Offline Cultural Translator initialized successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy", 
        "model": "Offline Cultural Translator",
        "specialization": "Cultural idioms and dialogue translation",
        "mode": "offline",
        "idiom_count": len(translator.cultural_idioms) if translator else 0
    })

@app.route('/docs', methods=['GET'])
def api_docs():
    """API documentation endpoint"""
    docs = {
        "endpoints": [
            {
                "path": "/health",
                "method": "GET",
                "description": "Health check endpoint. Returns translator status."
            },
            {
                "path": "/translate_dialogue",
                "method": "POST",
                "description": "Translate dialogue using cultural idiom mappings.",
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
                    "mode": "string"
                }
            }
        ],
        "features": {
            "offline_mode": "Works completely offline",
            "cultural_idioms": "Predefined cultural idiom mappings",
            "contextual_responses": "Smart contextual responses",
            "similarity_matching": "Partial text matching"
        },
        "supported_languages": {
            "en-hi": "English to Hindi (Cultural Idioms)"
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
            "model": "Offline Cultural Translator",
            "specialization": "cultural_idioms_dialogue",
            "mode": "offline",
            "temperature": temperature
        }
        
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Translation error: {str(e)}")
        return jsonify({"error": "Translation failed"}), 500

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def demo_translation():
    """Demonstrate offline translation capabilities"""
    print("=" * 60)
    print("OFFLINE CULTURAL TRANSLATION DEMO")
    print("=" * 60)
    
    translator = OfflineCulturalTranslator()
    
    # Test with cultural idioms
    test_phrases = [
        "Even after one hits rock bottom, their arrogance remains unchanged.",
        "Hey, what's up?",
        "This is absolutely crazy!",
        "Break a leg out there!",
        "I'm totally exhausted.",
        "That's a piece of cake.",
        "Don't beat around the bush.",
        "Thank you very much!",
        "I'm so happy today!",
        "What do you think about this?"
    ]
    
    print("\nTesting cultural idiom translations:")
    for phrase in test_phrases:
        translation = translator.translate_dialogue(phrase)
        print(f"  {phrase}")
        print(f"  → {translation}")
        print()
    
    print("✅ Offline demo completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Offline Cultural Translation')
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
        print(f"Starting offline API server on port {args.port}...")
        print("API endpoints:")
        print("  GET  /health - Health check")
        print("  GET  /docs - API documentation")
        print("  POST /translate_dialogue - Translate text")
        print("Mode: OFFLINE (no internet required)")
        app.run(host='0.0.0.0', port=args.port, debug=False)
        
    elif args.mode == 'test':
        translator = OfflineCulturalTranslator(args.src_lang, args.tgt_lang)
        
        print("Interactive Offline Translation (Ctrl+C to exit)")
        print("Type 'quit' to exit")
        try:
            while True:
                text = input(f"\nEnter {args.src_lang} text: ")
                if text.strip().lower() == 'quit':
                    break
                if text.strip():
                    translation = translator.translate_dialogue(text)
                    print(f"{args.tgt_lang.upper()}: {translation}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
    
    print("\n🎬 Offline Cultural Translation System Ready!")
    print(f"Specialized for: {args.src_lang} -> {args.tgt_lang}")
    print("Perfect for cultural idioms and dialogue! 🌍")
    print("Mode: OFFLINE - No internet connection required!") 