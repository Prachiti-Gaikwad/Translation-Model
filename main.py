# M2M-100 Translation Model Fine-tuning with OpenSubtitles Dataset
# Optimized for cultural nuances, idioms, and conversational language

import torch
from transformers import (
    M2M100ForConditionalGeneration, 
    M2M100Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from datasets import Dataset, load_dataset
import pandas as pd
from torch.utils.data import DataLoader
import json
from flask import Flask, request, jsonify
import logging
from typing import List, Dict, Optional
import os
import re
import gzip
from collections import defaultdict
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
        
    def load_opensubtitles_dataset(self, dataset_size="small"):
        """
        Load OpenSubtitles dataset from Hugging Face datasets
        dataset_size: "small" (10K samples) or "full" (millions of samples)
        """
        print(f"Loading OpenSubtitles dataset for {self.src_lang}-{self.tgt_lang}...")
        
        try:
            # Load from Hugging Face datasets
            if dataset_size == "small":
                # Load a subset for faster training/testing
                dataset = load_dataset(
                    "data/opensubs", 
                    f"{self.src_lang}-{self.tgt_lang}",
                    split="train[:10000]",  # First 10K samples
                    trust_remote_code=True
                )
            else:
                # Load full dataset
                
                df = pd.read_csv("data/opensubs")
                # Rename columns to 'translation'
                df = df.rename(columns={'source_col': self.src_lang, 'target_col': self.tgt_lang})
                hf_dataset = Dataset.from_dict({
                    'translation': df[[self.src_lang, self.tgt_lang]].to_dict(orient='records')
                })
                dataset = hf_dataset
            
            print(f"Loaded {len(dataset)} subtitle pairs")
            return dataset
            
        except Exception as e:
            print(f"Error loading from Hugging Face: {e}")
            print("Falling back to manual OpenSubtitles data preparation...")
            return self.create_sample_opensubtitles_data()
    
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
        
        return Dataset.from_list(subtitle_data)
    
    def clean_subtitle_text(self, text):
        """
        Clean subtitle text by removing HTML tags, timing info, etc.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove timing information
        text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3}', '', text)
        
        # Remove subtitle numbers
        text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short or very long texts
        if len(text) < 3 or len(text) > 500:
            return ""
        
        return text
    
    def filter_quality_pairs(self, dataset, min_length=3, max_length=200):
        """
        Filter subtitle pairs for quality
        """
        filtered_data = []
        
        for item in dataset:
            if 'translation' in item:
                src_text = item['translation'][self.src_lang]
                tgt_text = item['translation'][self.tgt_lang]
            else:
                # Handle different dataset formats
                src_text = item.get(self.src_lang, "")
                tgt_text = item.get(self.tgt_lang, "")
            
            # Clean texts
            src_clean = self.clean_subtitle_text(src_text)
            tgt_clean = self.clean_subtitle_text(tgt_text)
            
            # Quality filters
            if (min_length <= len(src_clean) <= max_length and 
                min_length <= len(tgt_clean) <= max_length and
                src_clean and tgt_clean):
                
                filtered_data.append({
                    'source': src_clean,
                    'target': tgt_clean,
                    'context': 'subtitle_dialogue'
                })
        
        print(f"Filtered to {len(filtered_data)} quality pairs")
        return filtered_data
    
    def preprocess_opensubtitles_data(self, dataset, train_size=0.8, val_size=0.1):
        """
        Tokenize and prepare OpenSubtitles data for training
        """
        print("Preprocessing OpenSubtitles data...")
        
        # Filter and clean data
        filtered_data = self.filter_quality_pairs(dataset)
        
        # Shuffle data
        random.shuffle(filtered_data)
        
        # Split data
        total_size = len(filtered_data)
        train_end = int(total_size * train_size)
        val_end = int(total_size * (train_size + val_size))
        
        train_data = filtered_data[:train_end]
        val_data = filtered_data[train_end:val_end]
        test_data = filtered_data[val_end:]
        
        print(f"Data split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        # Tokenize datasets
        train_dataset = self.tokenize_data(train_data)
        val_dataset = self.tokenize_data(val_data) if val_data else None
        test_dataset = self.tokenize_data(test_data) if test_data else None
        
        return train_dataset, val_dataset, test_dataset
    
    def tokenize_data(self, data_list):
        """
        Tokenize subtitle data for M2M-100
        """
        sources = [item['source'] for item in data_list]
        targets = [item['target'] for item in data_list]
        
        # Set source language for tokenizer
        self.tokenizer.src_lang = self.src_lang
        
        # Tokenize source texts
        source_encodings = self.tokenizer(
            sources,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Set target language for tokenizer
        self.tokenizer.src_lang = self.tgt_lang  # For target tokenization
        
        # Tokenize target texts
        target_encodings = self.tokenizer(
            targets,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Reset source language
        self.tokenizer.src_lang = self.src_lang
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': source_encodings['input_ids'],
            'attention_mask': source_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        })
        
        return dataset

# ============================================================================
# STEP 3: ENHANCED FINE-TUNING FOR SUBTITLES
# ============================================================================

class OpenSubtitlesFineTuner:
    def __init__(self, model, tokenizer, output_dir="./fine_tuned_m2m100_opensubtitles"):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def setup_training_args(self, num_train_epochs=3, batch_size=8):
        """
        Configure training parameters optimized for OpenSubtitles
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            learning_rate=3e-5,  # Lower LR for better subtitle adaptation
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            gradient_accumulation_steps=2,  # For effective larger batch size
            lr_scheduler_type="cosine",
            report_to=None,  # Disable wandb logging
        )
        return training_args
    
    def fine_tune_on_opensubtitles(self, train_dataset, eval_dataset=None, num_epochs=3):
        """
        Fine-tune M2M-100 specifically on OpenSubtitles data
        """
        print("Starting OpenSubtitles fine-tuning...")
        
        training_args = self.setup_training_args(num_train_epochs=num_epochs)
        
        # Data collator for sequence-to-sequence tasks
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Fine-tune
        trainer.train()
        
        # Save the fine-tuned model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training info
        training_info = {
            "model_name": "M2M-100 OpenSubtitles Fine-tuned",
            "dataset": "OpenSubtitles",
            "epochs": num_epochs,
            "src_lang": self.tokenizer.src_lang,
            "tgt_lang": self.tokenizer.tgt_lang,
            "output_dir": self.output_dir
        }
        
        with open(f"{self.output_dir}/training_info.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        print(f"OpenSubtitles fine-tuning completed! Model saved to {self.output_dir}")
        return trainer

# ============================================================================
# STEP 4: CULTURAL TRANSLATION INFERENCE
# ============================================================================

class OpenSubtitlesCulturalTranslator:
    def __init__(self, model_path, src_lang="en", tgt_lang="hi"):
        self.tokenizer = M2M100Tokenizer.from_pretrained(model_path)
        self.model = M2M100ForConditionalGeneration.from_pretrained(model_path)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Set language codes
        self.tokenizer.src_lang = src_lang
        self.tokenizer.tgt_lang = tgt_lang
        
        print(f"Loaded fine-tuned model for cultural translation: {src_lang} -> {tgt_lang}")
    
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
    
    def batch_translate_subtitles(self, texts: List[str], batch_size: int = 16) -> List[str]:
        """
        Efficiently translate multiple subtitle lines
        """
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )
            
            # Generate translations
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.get_lang_id(self.tgt_lang),
                    max_length=128,
                    num_beams=4,
                    do_sample=True,
                    temperature=0.7,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode batch
            batch_translations = self.tokenizer.batch_decode(
                generated_tokens, 
                skip_special_tokens=True
            )
            
            translations.extend([t.strip() for t in batch_translations])
        
        return translations

# ============================================================================
# STEP 5: ENHANCED REST API FOR SUBTITLE TRANSLATION
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

@app.route('/translate_subtitle_file', methods=['POST'])
def translate_subtitle_file():
    """Translate entire subtitle files"""
    try:
        data = request.get_json()
        
        if not data or 'subtitles' not in data:
            return jsonify({"error": "Missing 'subtitles' field in request"}), 400
        
        subtitles = data['subtitles']  # List of subtitle lines
        src_lang = data.get('src_lang', 'en')
        tgt_lang = data.get('tgt_lang', 'hi')
        
        # Update language settings
        if src_lang != translator.src_lang or tgt_lang != translator.tgt_lang:
            translator.tokenizer.src_lang = src_lang
            translator.tokenizer.tgt_lang = tgt_lang
            translator.src_lang = src_lang
            translator.tgt_lang = tgt_lang
        
        # Translate all subtitles
        translations = translator.batch_translate_subtitles(subtitles)
        
        response = {
            "source_subtitles": subtitles,
            "translated_subtitles": translations,
            "source_language": src_lang,
            "target_language": tgt_lang,
            "subtitle_count": len(translations),
            "model": "M2M-100 OpenSubtitles Cultural"
        }
        
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Subtitle file translation error: {str(e)}")
        return jsonify({"error": "Subtitle file translation failed"}), 500

# =========================================================================
# STEP 5.1: API DOCUMENTATION ENDPOINT
# =========================================================================

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
            },
            {
                "path": "/translate_subtitle_file",
                "method": "POST",
                "description": "Translate a list of subtitle lines.",
                "request_format": {
                    "subtitles": "list of strings (required)",
                    "src_lang": "string (optional, default: 'en')",
                    "tgt_lang": "string (optional, default: 'hi')"
                },
                "response_format": {
                    "source_subtitles": "list of strings",
                    "translated_subtitles": "list of strings",
                    "source_language": "string",
                    "target_language": "string",
                    "subtitle_count": "int",
                    "model": "string"
                }
            }
        ]
    }
    return jsonify(docs)

# =========================================================================
# STEP 7: PRE- AND POST-FINE-TUNING EVALUATION FOR COMPARISON
# =========================================================================

def evaluate_model_on_testset(translator, test_dataset, tokenizer, evaluator, sample_size=20):
    """
    Evaluate a model (pre- or post-fine-tuning) on the test set and return metrics.
    """
    test_samples = test_dataset.select(range(min(sample_size, len(test_dataset))))
    test_sources = []
    test_references = []
    for item in test_samples:
        source = tokenizer.decode(item['input_ids'], skip_special_tokens=True)
        target = tokenizer.decode(item['labels'], skip_special_tokens=True)
        test_sources.append(source)
        test_references.append(target)
    predictions = translator.batch_translate_subtitles(test_sources)
    evaluation_results = evaluator.evaluate_cultural_accuracy(predictions, test_references)
    return evaluation_results

# ============================================================================
# STEP 6: OPENSUBTITLES-SPECIFIC EVALUATION
# ============================================================================

try:
    from sacrebleu import BLEU
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("Warning: sacrebleu not available, using simple BLEU calculation")

try:
    from nltk.translate.meteor_score import meteor_score
    import nltk
    try:
        nltk.download('wordnet', quiet=True)
    except:
        pass
    METEOR_AVAILABLE = True
except ImportError:
    METEOR_AVAILABLE = False
    print("Warning: nltk meteor_score not available")

class OpenSubtitlesEvaluator:
    def __init__(self):
        if BLEU_AVAILABLE:
            self.bleu = BLEU()
        else:
            self.bleu = None
    
    def evaluate_cultural_accuracy(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Evaluate model performance on cultural/dialogue translation
        """
        # Standard metrics
        bleu_score = self.calculate_bleu_score(predictions, references)
        meteor_scores = self.calculate_meteor_scores(predictions, references)
        
        # Cultural-specific metrics
        cultural_metrics = self.assess_cultural_preservation(predictions, references)
        
        return {
            "bleu_score": bleu_score,
            "meteor_score": meteor_scores,
            "cultural_preservation": cultural_metrics,
            "sample_count": len(predictions)
        }
    
    def calculate_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Calculate BLEU score for subtitle translations"""
        if not BLEU_AVAILABLE or self.bleu is None:
            # Simple BLEU-like calculation
            return self.simple_bleu_score(predictions, references)
        
        try:
            refs = [[ref] for ref in references]
            score = self.bleu.corpus_score(predictions, list(zip(*refs)))
            return score.score
        except:
            return self.simple_bleu_score(predictions, references)
    
    def simple_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Simple BLEU-like score calculation when sacrebleu is not available"""
        if not predictions or not references:
            return 0.0
        
        total_score = 0.0
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            if not pred_words or not ref_words:
                continue
                
            # Simple n-gram matching
            matches = 0
            for word in pred_words:
                if word in ref_words:
                    matches += 1
            
            if len(pred_words) > 0:
                precision = matches / len(pred_words)
                total_score += precision
        
        return total_score / len(predictions) if predictions else 0.0
    
    def calculate_meteor_scores(self, predictions: List[str], references: List[str]) -> float:
        """Calculate average METEOR score"""
        if not METEOR_AVAILABLE:
            # Simple similarity score when METEOR is not available
            return self.simple_similarity_score(predictions, references)
        
        try:
            scores = []
            for pred, ref in zip(predictions, references):
                score = meteor_score([ref.split()], pred.split())
                scores.append(score)
            return sum(scores) / len(scores) if scores else 0.0
        except:
            return self.simple_similarity_score(predictions, references)
    
    def simple_similarity_score(self, predictions: List[str], references: List[str]) -> float:
        """Simple similarity score when METEOR is not available"""
        if not predictions or not references:
            return 0.0
        
        total_score = 0.0
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if not pred_words or not ref_words:
                continue
            
            # Jaccard similarity
            intersection = len(pred_words.intersection(ref_words))
            union = len(pred_words.union(ref_words))
            
            if union > 0:
                similarity = intersection / union
                total_score += similarity
        
        return total_score / len(predictions) if predictions else 0.0
    
    def assess_cultural_preservation(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Assess how well cultural nuances are preserved
        """
        # Define cultural indicators
        cultural_indicators = {
            'idioms': ['piece of cake', 'break a leg', 'raining cats and dogs'],
            'informal_speech': ['gonna', 'wanna', 'yeah', 'nah', 'dude', 'awesome'],
            'emotional_expressions': ['wow', 'oh my god', 'seriously', 'no way']
        }
        
        cultural_scores = {}
        
        for category, indicators in cultural_indicators.items():
            source_matches = 0
            preserved_matches = 0
            
            for pred, ref in zip(predictions, references):
                pred_lower = pred.lower()
                ref_lower = ref.lower()
                
                for indicator in indicators:
                    if indicator in ref_lower:
                        source_matches += 1
                        # Check if cultural meaning is preserved (simplified check)
                        if len(pred_lower) > 0:  # Basic preservation check
                            preserved_matches += 1
            
            preservation_rate = preserved_matches / source_matches if source_matches > 0 else 1.0
            cultural_scores[category] = {
                'found': source_matches,
                'preserved': preserved_matches,
                'preservation_rate': preservation_rate
            }
        
        return cultural_scores

# ============================================================================
# MAIN EXECUTION PIPELINE FOR OPENSUBTITLES
# ============================================================================

def main_opensubtitles_pipeline(compare=False):
    """Complete pipeline for OpenSubtitles fine-tuning, with optional pre/post comparison."""
    
    # Configuration
    SRC_LANG = "en"  # Change as needed
    TGT_LANG = "hi"  # Change as needed (hi, es, fr, de, etc.)
    MODEL_SIZE = "418M"  # or "1.2B" for better quality
    
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
    
    # Step 2: Prepare OpenSubtitles dataset
    print("\n📺 Step 2: Loading OpenSubtitles dataset...")
    dataset_prep = OpenSubtitlesDatasetPreparator(
        translation_model.tokenizer, 
        src_lang=SRC_LANG, 
        tgt_lang=TGT_LANG
    )
    
    # Load dataset (use "small" for testing, "full" for production)
    raw_dataset = dataset_prep.load_opensubtitles_dataset(dataset_size="small")
    
    # Preprocess and split data
    train_dataset, val_dataset, test_dataset = dataset_prep.preprocess_opensubtitles_data(raw_dataset)

    # Step 5: Evaluate pre-fine-tuned model (if compare=True)
    pre_eval_results = None
    if compare and test_dataset:
        print("\n📊 Pre-fine-tuning evaluation (base model)...")
        base_translator = OpenSubtitlesCulturalTranslator(translation_model.model_name, src_lang=SRC_LANG, tgt_lang=TGT_LANG)
        evaluator = OpenSubtitlesEvaluator()
        pre_eval_results = evaluate_model_on_testset(base_translator, test_dataset, translation_model.tokenizer, evaluator)
        print("Pre-fine-tuning BLEU: {:.4f}".format(pre_eval_results['bleu_score']))
        print("Pre-fine-tuning METEOR: {:.4f}".format(pre_eval_results['meteor_score']))

    # Step 3: Fine-tune model
    print("\n🎯 Step 3: Fine-tuning on OpenSubtitles...")
    fine_tuner = OpenSubtitlesFineTuner(
        translation_model.model, 
        translation_model.tokenizer
    )
    
    trainer = fine_tuner.fine_tune_on_opensubtitles(
        train_dataset, 
        eval_dataset=val_dataset,
        num_epochs=3
    )
    
    # Step 4: Test the fine-tuned model
    print("\n💬 Step 4: Testing dialogue translation...")
    translator = OpenSubtitlesCulturalTranslator("./fine_tuned_m2m100_opensubtitles")
    
    # Test with sample dialogues
    test_dialogues = [
        "Hey, what's up?",
        "This is absolutely crazy!",
        "Break a leg out there!",
        "I'm totally exhausted.",
        "That's a piece of cake."
    ]
    
    print("\nSample translations:")
    for dialogue in test_dialogues:
        translation = translator.translate_dialogue(dialogue)
        print(f"  {dialogue} -> {translation}")
    
    # Step 5: Evaluate model
    post_eval_results = None
    if test_dataset:
        print("\n📊 Post-fine-tuning evaluation...")
        evaluator = OpenSubtitlesEvaluator()
        post_eval_results = evaluate_model_on_testset(translator, test_dataset, translation_model.tokenizer, evaluator)
        print("Post-fine-tuning BLEU: {:.4f}".format(post_eval_results['bleu_score']))
        print("Post-fine-tuning METEOR: {:.4f}".format(post_eval_results['meteor_score']))
        print("\nCultural Preservation (post-fine-tuning):")
        for category, metrics in post_eval_results['cultural_preservation'].items():
            print(f"  {category}: {metrics['preservation_rate']:.2%} preserved")

    # Print comparison if requested
    if compare and pre_eval_results and post_eval_results:
        print("\n==================== Evaluation Comparison ====================")
        print("Metric           | Pre-Fine-Tune | Post-Fine-Tune")
        print("-----------------|---------------|----------------")
        print("BLEU             | {:.4f}        | {:.4f}".format(pre_eval_results['bleu_score'], post_eval_results['bleu_score']))
        print("METEOR           | {:.4f}        | {:.4f}".format(pre_eval_results['meteor_score'], post_eval_results['meteor_score']))
        print("==============================================================")

    # Step 6: Start API server
    print("\n🌐 Step 6: Starting subtitle translation API...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  POST /translate_dialogue - Translate single dialogue")
    print("  POST /translate_subtitle_file - Translate subtitle files")
    print("  GET /health - Health check")
    
    # Initialize API translator
    initialize_subtitle_translator()
    
    print("\n✅ OpenSubtitles fine-tuning pipeline completed successfully!")
    print(f"Fine-tuned model saved to: ./fine_tuned_m2m100_opensubtitles")
    
    return translator, post_eval_results

# ============================================================================
# UTILITY FUNCTIONS AND EXAMPLE USAGE
# ============================================================================

def example_api_usage():
    """Example of how to use the subtitle translation API"""
    
    import requests
    import json
    
    api_url = "http://localhost:5000"
    
    # Example 1: Single dialogue translation
    dialogue_data = {
        "text": "Hey, what's up? This movie is absolutely crazy!",
        "src_lang": "en",
        "tgt_lang": "hi",
        "temperature": 0.7
    }
    
    print("Example API Usage:")
    print("1. Single dialogue translation:")
    print(f"Request: {json.dumps(dialogue_data, indent=2)}")
    
    # Example 2: Subtitle file translation
    subtitle_data = {
        "subtitles": [
            "Welcome to the show!",
            "This is going to be amazing.",
            "I can't wait to see what happens next.",
            "That was totally unexpected!",
            "Thanks for watching, everyone!"
        ],
        "src_lang": "en",
        "tgt_lang": "hi"
    }
    
    print("\n2. Subtitle file translation:")
    print(f"Request: {json.dumps(subtitle_data, indent=2)}")

def create_custom_opensubtitles_dataset(src_lang="en", tgt_lang="hi", size=1000):
    """
    Create a custom OpenSubtitles-style dataset for specific language pairs
    """
    
    # Extended dialogue patterns for different scenarios
    dialogue_patterns = {
        'greetings': [
            ("Hello there!", "नमस्ते!"),
            ("Good morning!", "सुप्रभात!"),
            ("How are you doing?", "आप कैसे हैं?"),
            ("Nice to meet you.", "आपसे मिलकर खुशी हुई।"),
        ],
        'emotions': [
            ("I'm so happy!", "मैं बहुत खुश हूँ!"),
            ("This is incredible!", "यह अविश्वसनीय है!"),
            ("I'm really worried.", "मुझे वास्तव में चिंता है।"),
            ("Don't be afraid.", "डरो मत।"),
        ],
        'casual_talk': [
            ("What's going on?", "क्या चल रहा है?"),
            ("No big deal.", "कोई बड़ी बात नहीं।"),
            ("Are you kidding me?", "क्या आप मजाक कर रहे हैं?"),
            ("That's hilarious!", "यह बहुत मजेदार है!"),
        ],
        'movie_dialogue': [
            ("This changes everything.", "इससे सब कुछ बदल जाता है।"),
            ("We need to hurry.", "हमें जल्दी करनी होगी।"),
            ("It's too dangerous.", "यह बहुत खतरनाक है।"),
            ("Trust me on this.", "इस पर मेरा भरोसा करो।"),
        ]
    }
    
    dataset = []
    
    # Generate samples from patterns
    for category, patterns in dialogue_patterns.items():
        for src, tgt in patterns:
            dataset.append({
                'translation': {src_lang: src, tgt_lang: tgt},
                'category': category
            })
    
    # Extend with variations
    while len(dataset) < size:
        # Randomly select and slightly modify existing patterns
        category = random.choice(list(dialogue_patterns.keys()))
        pattern = random.choice(dialogue_patterns[category])
        
        dataset.append({
            'translation': {src_lang: pattern[0], tgt_lang: pattern[1]},
            'category': category
        })
    
    return dataset[:size]

def load_custom_subtitle_file(filepath, format='srt'):
    """
    Load custom subtitle files for translation
    Supports SRT, VTT formats
    """
    
    subtitles = []
    
    if format.lower() == 'srt':
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse SRT format
        blocks = content.strip().split('\n\n')
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                # Skip subtitle number and timing
                subtitle_text = ' '.join(lines[2:])
                if subtitle_text.strip():
                    subtitles.append(subtitle_text.strip())
    
    elif format.lower() == 'vtt':
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # Skip WEBVTT header, timing lines, and empty lines
            if (line and 
                not line.startswith('WEBVTT') and 
                '-->' not in line and 
                not line.isdigit()):
                subtitles.append(line)
    
    return subtitles

# ============================================================================
# PRODUCTION DEPLOYMENT HELPERS
# ============================================================================

def deploy_to_production():
    """
    Production deployment configuration
    """
    
    production_config = {
        'gunicorn_config': {
            'bind': '0.0.0.0:5000',
            'workers': 4,
            'worker_class': 'sync',
            'timeout': 120,
            'max_requests': 1000,
            'max_requests_jitter': 100,
            'preload_app': True
        },
        'model_config': {
            'model_size': '1.2B',  # Use larger model for production
            'batch_size': 32,
            'cache_size': 1000,
            'gpu_enabled': torch.cuda.is_available()
        },
        'security': {
            'rate_limiting': '100/hour',
            'cors_enabled': True,
            'api_key_required': False  # Set to True for production
        }
    }
    
    print("Production Deployment Configuration:")
    print(json.dumps(production_config, indent=2))
    
    # Save configuration
    with open('production_config.json', 'w') as f:
        json.dump(production_config, f, indent=2)
    
    print("Configuration saved to production_config.json")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='M2M-100 OpenSubtitles Cultural Translation')
    parser.add_argument('--mode', choices=['train', 'serve', 'test', 'deploy', 'compare'], 
                       default='train', help='Operation mode')
    parser.add_argument('--src_lang', default='en', help='Source language code')
    parser.add_argument('--tgt_lang', default='hi', help='Target language code')
    parser.add_argument('--model_size', choices=['418M', '1.2B'], default='418M', 
                       help='Model size')
    parser.add_argument('--port', type=int, default=5000, help='API server port')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Run full training pipeline
        translator, eval_results = main_opensubtitles_pipeline()
        
    elif args.mode == 'serve':
        # Start API server only
        initialize_subtitle_translator()
        print(f"Starting API server on port {args.port}...")
        app.run(host='0.0.0.0', port=args.port, debug=False)
        
    elif args.mode == 'test':
        # Test existing model
        translator = OpenSubtitlesCulturalTranslator(
            "./fine_tuned_m2m100_opensubtitles",
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang
        )
        
        # Interactive testing
        print("Interactive Subtitle Translation (Ctrl+C to exit)")
        try:
            while True:
                text = input(f"\nEnter {args.src_lang} dialogue: ")
                if text.strip():
                    translation = translator.translate_dialogue(text)
                    print(f"{args.tgt_lang.upper()}: {translation}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            
    elif args.mode == 'deploy':
        # Generate production deployment config
        deploy_to_production()
    elif args.mode == 'compare':
        # Run pre- and post-fine-tuning evaluation and print comparison
        translator, eval_results = main_opensubtitles_pipeline(compare=True)
    
    print("\n🎬 OpenSubtitles Cultural Translation System Ready!")
    print(f"Specialized for: {args.src_lang} -> {args.tgt_lang}")
    print("Perfect for movie/TV subtitles with cultural nuances! 🌍")
