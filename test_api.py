import requests
import json
import pandas as pd
import ast
import os
from pathlib import Path

def load_csv_data(csv_path):
    """Load translation data from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} entries from {csv_path}")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

def extract_translations_from_row(row):
    """Extract source and target text from a CSV row"""
    try:
        # Parse the translation dictionary string
        translation_dict = ast.literal_eval(row['translation'])
        
        # Get language codes from the dictionary keys
        languages = list(translation_dict.keys())
        
        if len(languages) >= 2:
            source_lang = languages[0]  # First language as source
            target_lang = languages[1]  # Second language as target
            
            source_text = translation_dict[source_lang]
            target_text = translation_dict[target_lang]
            
            return {
                'source_lang': source_lang,
                'target_lang': target_lang,
                'source_text': source_text,
                'target_text': target_text
            }
    except Exception as e:
        print(f"❌ Error parsing translation row: {e}")
        return None

def test_translation_api():
    """Test the translation API with idioms from CSV files"""
    url = "http://localhost:5000/translate_dialogue"
    
    # Get all CSV files from the data directory
    data_dir = Path("data/opensubs")
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found in data/opensubs directory")
        return
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    # Test with the first few entries from each CSV file
    for csv_file in csv_files[:3]:  # Limit to first 3 files to avoid too many requests
        print(f"\n📄 Testing with {csv_file.name}")
        print("=" * 50)
        
        # Load CSV data
        df = load_csv_data(csv_file)
        if df is None:
            continue
        
        # Test with first 5 entries from this file
        for index, row in df.head(5).iterrows():
            translation_data = extract_translations_from_row(row)
            
            if translation_data is None:
                continue
            
            source_text = translation_data['source_text']
            source_lang = translation_data['source_lang']
            target_lang = translation_data['target_lang']
            
            # Skip if source text is too short or contains special characters
            if len(source_text) < 10 or len(source_text) > 200:
                continue
            
            print(f"\n🔄 Testing translation {index + 1}:")
            print(f"Source ({source_lang}): {source_text[:100]}...")
            
            payload = {
                "text": source_text,
                "source_lang": source_lang,
                "target_lang": target_lang
            }
            headers = {"Content-Type": "application/json"}
            
            try:
                response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    result = response.json()
                    translation = result.get('translation', 'N/A')
                    print(f"✅ Translation: {translation[:100]}...")
                    
                    # Compare with original target text if available
                    original_target = translation_data['target_text']
                    print(f"📖 Original: {original_target[:100]}...")
                    
                else:
                    print(f"❌ Error {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"❌ Connection Error: {e}")
            
            # Add a small delay between requests
            import time
            time.sleep(0.5)

def test_specific_idioms():
    """Test with specific cultural idioms"""
    url = "http://localhost:5000/translate_dialogue"
    
    idioms = [
        "Even after one hits rock bottom, their arrogance remains unchanged.",
        "The early bird catches the worm.",
        "Actions speak louder than words.",
        "Don't judge a book by its cover.",
        "When in Rome, do as the Romans do."
    ]
    
    print("\n🎯 Testing specific cultural idioms:")
    print("=" * 50)
    
    for i, idiom in enumerate(idioms, 1):
        print(f"\n🔄 Idiom {i}: {idiom}")
        
        payload = {"text": idiom}
        headers = {"Content-Type": "application/json"}
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                translation = result.get('translation', 'N/A')
                print(f"✅ Translation: {translation}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    print("🚀 Starting Translation API Testing")
    print("=" * 60)
    
    # Test with CSV data
    test_translation_api()
    
    # Test with specific idioms
    test_specific_idioms()
    
    print("\n✨ Testing completed!") 
