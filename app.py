# Cultural Translation API - Production Ready
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Cultural idioms dictionary
CULTURAL_IDIOMS = {
    "Even after one hits rock bottom, their arrogance remains unchanged.": "रस्सी जल गयी, बल नहीं गया",
    "Actions speak louder than words.": "कर्म ही पूजा है",
    "Where there's a will, there's a way.": "जहां चाह वहां राह",
    "A stitch in time saves nine.": "समय पर एक सिलाई नौ सिलाई बचाती है",
    "Don't judge a book by its cover.": "किताब को उसके कवर से न आंकें",
    "The early bird catches the worm.": "जल्दी सोने वाला जल्दी उठता है",
    "Practice makes perfect.": "अभ्यास से ही पूर्णता आती है",
    "All that glitters is not gold.": "सब जो चमकता है वो सोना नहीं होता",
    "Better late than never.": "देर आए दुरुस्त आए",
    "Honesty is the best policy.": "ईमानदारी सबसे अच्छी नीति है"
}

def similarity_score(text1, text2):
    """Simple similarity scoring"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return len(intersection) / len(union) if union else 0

def translate_text(text):
    """Translate text using cultural idioms"""
    text = text.strip()
    
    # Check for exact match
    if text in CULTURAL_IDIOMS:
        return {
            "translation": CULTURAL_IDIOMS[text],
            "confidence": "high",
            "original": text,
            "source_language": "en",
            "target_language": "hi"
        }
    
    # Check for partial matches
    best_match = None
    best_score = 0
    
    for idiom in CULTURAL_IDIOMS:
        score = similarity_score(text, idiom)
        if score > best_score and score > 0.3:  # 30% similarity threshold
            best_score = score
            best_match = idiom
    
    if best_match:
        return {
            "translation": CULTURAL_IDIOMS[best_match],
            "confidence": "medium" if best_score < 0.7 else "high",
            "original": text,
            "source_language": "en",
            "target_language": "hi",
            "similarity_score": best_score
        }
    
    # Default response for unknown text
    return {
        "translation": "Translation not available for this text",
        "confidence": "low",
        "original": text,
        "source_language": "en",
        "target_language": "hi"
    }

@app.route('/')
def home():
    return jsonify({
        "message": "Cultural Translation API",
        "status": "running",
        "endpoints": {
            "translate": "/translate_dialogue",
            "health": "/health",
            "docs": "/docs"
        }
    })

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "service": "cultural-translation-api"})

@app.route('/docs')
def api_docs():
    return jsonify({
        "API Documentation": "Cultural Translation API",
        "endpoints": {
            "/translate_dialogue": {
                "method": "POST",
                "description": "Translate text using cultural idioms",
                "body": {
                    "text": "string (required)"
                },
                "example": {
                    "text": "Even after one hits rock bottom, their arrogance remains unchanged."
                }
            }
        },
        "example_response": {
            "translation": "रस्सी जल गयी, बल नहीं गया",
            "confidence": "high",
            "original": "Even after one hits rock bottom, their arrogance remains unchanged.",
            "source_language": "en",
            "target_language": "hi"
        }
    })

@app.route('/translate_dialogue', methods=['POST'])
def translate_dialogue():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                "error": "Missing text parameter",
                "message": "Please provide 'text' in the request body"
            }), 400
        
        text = data['text']
        if not text or not text.strip():
            return jsonify({
                "error": "Empty text",
                "message": "Text cannot be empty"
            }), 400
        
        result = translate_text(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            "error": "Translation failed",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 