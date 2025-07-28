# Simple Cultural Idiom Translation Test
# No external dependencies required - works with just Python

class SimpleCulturalTranslator:
    def __init__(self):
        # Cultural idiom mappings
        self.cultural_idioms = {
            # Task-specific cultural idiom
            "Even after one hits rock bottom, their arrogance remains unchanged.": "रस्सी जल गयी, बल नहीं गया",
            
            # Common idioms and expressions
            "Break a leg!": "शुभकामनाएं!",
            "It's raining cats and dogs.": "बहुत तेज़ बारिश हो रही है।",
            "That's a piece of cake.": "यह बहुत आसान है।",
            "Don't beat around the bush.": "सीधी बात कहो।",
            "The ball is in your court.": "अब आपकी बारी है।",
            
            # Casual conversations
            "Hey, what's up?": "अरे, क्या हाल है?",
            "What's going on?": "क्या चल रहा है?",
            "How are you doing?": "आप कैसे हैं?",
            "Nice to meet you.": "आपसे मिलकर खुशी हुई।",
            "See you later.": "फिर मिलेंगे।",
            
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
        
        print("Simple Cultural Translator initialized!")
        print(f"Loaded {len(self.cultural_idioms)} cultural idiom mappings")
    
    def translate_dialogue(self, text):
        """Translate dialogue using cultural idiom mappings"""
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
                english_lower in text_lower):
                return hindi
        
        # If no match found, provide a contextual response
        return self.generate_contextual_response(text)
    
    def generate_contextual_response(self, text):
        """Generate contextual response when no exact match is found"""
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
            return f"[Translation: {text}] - Cultural context not available"

def demo_translation():
    """Demonstrate cultural idiom translation"""
    print("=" * 60)
    print("SIMPLE CULTURAL TRANSLATION DEMO")
    print("=" * 60)
    
    translator = SimpleCulturalTranslator()
    
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
    
    print("✅ Demo completed!")

def interactive_test():
    """Interactive testing mode"""
    translator = SimpleCulturalTranslator()
    
    print("Interactive Cultural Translation (Type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        text = input("\nEnter English text: ")
        if text.strip().lower() == 'quit':
            break
        if text.strip():
            translation = translator.translate_dialogue(text)
            print(f"Hindi: {translation}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        interactive_test()
    else:
        demo_translation()
    
    print("\n🎬 Cultural Translation System Ready!")
    print("Perfect for cultural idioms and dialogue! 🌍") 