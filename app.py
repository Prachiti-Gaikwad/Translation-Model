# Cultural Translation API - Minimal Version
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Cultural idioms dictionary
CULTURAL_IDIOMS = {
    "Even after one hits rock bottom, their arrogance remains unchanged.": "रस्सी जल गयी, बल नहीं गया",
    "Actions speak louder than words.": "कर्म ही पूजा है",
    "Where there's a will, there's a way.": "जहां चाह वहां राह",
    "A stitch in time saves nine.": "समय पर एक सिलाई नौ सिलाई बचाती है",
    "Don't judge a book by its cover.": "किताब को उसके कवर से न आंकें"
}

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
    
    # Default response
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
            "health": "/health"
        }
    })

@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "service": "cultural-translation-api"})

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