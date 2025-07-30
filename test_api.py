import requests
import json

def test_translation_api():
    url = "http://localhost:5000/translate_dialogue"
    
    # Test the cultural idiom
    test_text = "Even after one hits rock bottom, their arrogance remains unchanged."
    
    payload = {"text": test_text}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ SUCCESS!")
            print(f"Input: {test_text}")
            print(f"Translation: {result.get('translation', 'N/A')}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_translation_api() 