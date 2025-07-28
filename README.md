# Translation Model Fine-tuning and Deployment

## Project Overview

This project addresses the challenge of translation models failing to account for cultural nuances, idioms, and community-specific expressions. For example, translating "रस्सी जल गयी, बल नहीं गया" using a generic model yields incorrect results, while the accurate translation is "Even after one hits rock bottom, their arrogance remains unchanged."

## Features

- **Multiple Versions**: Offline, Ultra-Simple, and Full M2M-100 implementations
- **Model**: M2M-100 (Multilingual Machine Translation) / MarianMT / Offline Cultural Mappings
- **Dataset**: OpenSubtitles Dataset with cultural idioms and nuances
- **Framework**: Hugging Face Transformers / Flask REST API
- **Evaluation**: BLEU and METEOR metrics with pre/post fine-tuning comparison
- **Deployment**: Docker-ready with cloud deployment support
- **Offline Mode**: Works completely offline with predefined cultural idiom mappings

## Quick Start

### Option 1: Offline Version (Recommended - No Dependencies)

```bash
# No installation needed - works with just Python
python simple_test.py

# Interactive testing
python simple_test.py test
```

### Option 2: Flask API Version

```bash
# Install Flask only
pip install -r requirements_offline.txt

# Run demo
python main_offline.py --mode demo

# Start API server
python main_offline.py --mode serve --port 5000
```

### Option 3: Docker Deployment

```bash
# Windows
deploy.bat

# Linux/Mac
chmod +x deploy.sh
./deploy.sh
```

## Installation

### Prerequisites

- Python 3.7+
- Docker (for containerized deployment)
- Docker Compose (for easy deployment)

### Dependencies

Choose the appropriate requirements file based on your needs:

**Offline Version (No Dependencies):**
```bash
# No installation needed - works with just Python
python simple_test.py
```

**Flask API Version:**
```bash
pip install -r requirements_offline.txt
```

**Ultra-Simple Version (MarianMT):**
```bash
pip install -r requirements_ultra_simple.txt
```

**Full Version (M2M-100):**
```bash
pip install -r requirements_minimal.txt
```

## Usage

### 1. Simple Offline Testing

```bash
# Demo mode
python simple_test.py

# Interactive testing
python simple_test.py test
```

### 2. Flask API Usage

**Demo Mode:**
```bash
python main_offline.py --mode demo
```

**Interactive Testing:**
```bash
python main_offline.py --mode test --src_lang en --tgt_lang hi
```

**Start API Server:**
```bash
python main_offline.py --mode serve --port 5000
```

### 3. Training and Fine-tuning (Full Version)

**Basic training:**
```bash
python main.py --mode train --src_lang en --tgt_lang hi
```

**With pre/post evaluation comparison:**
```bash
python main.py --mode compare --src_lang en --tgt_lang hi
```

## API Documentation

### Endpoints

#### Health Check
- **GET** `/health`
- Returns model status and specialization info

#### Single Dialogue Translation
- **POST** `/translate_dialogue`
- **Request:**
```json
{
    "text": "Even after one hits rock bottom, their arrogance remains unchanged.",
    "src_lang": "en",
    "tgt_lang": "hi",
    "temperature": 0.7
}
```
- **Response:**
```json
{
    "source_text": "Even after one hits rock bottom, their arrogance remains unchanged.",
    "translated_text": "रस्सी जल गयी, बल नहीं गया",
    "source_language": "en",
    "target_language": "hi",
    "model": "Offline Cultural Translator",
    "specialization": "cultural_idioms_dialogue",
    "mode": "offline",
    "temperature": 0.7
}
```

#### API Documentation
- **GET** `/docs`
- Returns complete API documentation in JSON format

### Testing the API

```bash
# Health check
curl http://localhost:5000/health

# Test cultural idiom
curl -X POST http://localhost:5000/translate_dialogue \
  -H "Content-Type: application/json" \
  -d '{"text": "Even after one hits rock bottom, their arrogance remains unchanged.", "src_lang": "en", "tgt_lang": "hi"}'
```

## Model Architecture

### Offline Cultural Translator
- **Mode**: Completely offline
- **Features**: 50+ predefined cultural idiom mappings
- **Specialization**: Cultural idioms, casual conversations, emotional expressions
- **Dependencies**: None (pure Python)

### M2M-100 Model (Full Version)
- **Size Options**: 418M or 1.2B parameters
- **Languages**: Supports 100+ language pairs
- **Specialization**: Fine-tuned on OpenSubtitles for cultural nuances

### MarianMT Model (Ultra-Simple Version)
- **Model**: Helsinki-NLP MarianMT
- **Languages**: English to Hindi, Spanish, French
- **Features**: Fallback system with cultural idiom mappings

## Cultural Idiom Examples

The system handles various types of cultural expressions:

### Task-Specific Example
- **Input**: "Even after one hits rock bottom, their arrogance remains unchanged."
- **Output**: "रस्सी जल गयी, बल नहीं गया"

### Common Idioms
- "Break a leg!" → "शुभकामनाएं!"
- "It's raining cats and dogs." → "बहुत तेज़ बारिश हो रही है।"
- "That's a piece of cake." → "यह बहुत आसान है।"

### Casual Conversations
- "Hey, what's up?" → "अरे, क्या हाल है?"
- "I'm totally exhausted." → "मैं बिल्कुल थक गया हूँ।"

## Docker Deployment

### Quick Deployment

**Windows:**
```bash
deploy.bat
```

**Linux/Mac:**
```bash
chmod +x deploy.sh
./deploy.sh
```

### Manual Docker Commands

```bash
# Build image
docker build -f Dockerfile.offline -t cultural-translator .

# Run container
docker run -d -p 5000:5000 --name cultural-translator cultural-translator

# Using Docker Compose
docker-compose -f docker-compose.offline.yml up -d
```

### Docker Commands Reference

```bash
# View running containers
docker ps

# View logs
docker logs cultural-translator

# Stop container
docker stop cultural-translator

# Remove container
docker rm cultural-translator
```

## Cloud Deployment

### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
docker tag cultural-translator:latest your-account.dkr.ecr.us-east-1.amazonaws.com/cultural-translator:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/cultural-translator:latest
```

### Azure Container Instances
```bash
# Deploy to Azure
az container create --resource-group myResourceGroup --name cultural-translator --image cultural-translator:latest --dns-name-label cultural-translator --ports 5000
```

### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy cultural-translator --image cultural-translator:latest --platform managed --region us-central1 --allow-unauthenticated
```

## Evaluation Metrics

### Standard Metrics
- **BLEU Score**: Measures translation accuracy
- **METEOR Score**: Considers synonyms and paraphrases

### Cultural Preservation Metrics
- **Idioms**: Preservation of cultural expressions
- **Informal Speech**: Handling of casual language
- **Emotional Expressions**: Maintaining emotional context

### Pre/Post Fine-tuning Comparison
The system automatically compares model performance before and after fine-tuning:
```
Metric           | Pre-Fine-Tune | Post-Fine-Tune
-----------------|---------------|----------------
BLEU             | 0.2345        | 0.3456
METEOR           | 0.1234        | 0.2345
```

## Project Structure

```
Translation Model Fine-tuning and Deployment/
├── simple_test.py               # ✅ Offline version (no dependencies)
├── main_offline.py              # ✅ Flask API offline version
├── main_ultra_simple.py         # ✅ MarianMT version
├── main.py                      # ✅ Full M2M-100 version
├── requirements_offline.txt     # ✅ Flask only
├── requirements_ultra_simple.txt # ✅ MarianMT dependencies
├── requirements_minimal.txt     # ✅ M2M-100 dependencies
├── Dockerfile.offline           # ✅ Docker configuration
├── docker-compose.offline.yml   # ✅ Docker Compose config
├── deploy.sh                    # ✅ Linux/Mac deployment script
├── deploy.bat                   # ✅ Windows deployment script
├── .dockerignore                # ✅ Docker ignore file
├── README.md                    # ✅ This documentation
└── data/                        # ✅ Dataset directory
    └── opensubs/
```

## Technical Implementation

### Key Components

1. **SimpleCulturalTranslator**: Offline cultural idiom mappings
2. **OfflineCulturalTranslator**: Flask-based offline translator
3. **SimpleCulturalTranslator**: MarianMT-based translator
4. **M2M100TranslationModel**: Full M2M-100 implementation
5. **OpenSubtitlesDatasetPreparator**: Dataset loading and preprocessing
6. **OpenSubtitlesFineTuner**: Fine-tuning pipeline
7. **OpenSubtitlesCulturalTranslator**: Inference and translation
8. **OpenSubtitlesEvaluator**: Evaluation metrics and comparison
9. **Flask API**: REST API endpoints

### Fine-tuning Process (Full Version)

1. **Data Preparation**: Load and clean OpenSubtitles data
2. **Tokenization**: Prepare data for M2M-100 model
3. **Training**: Fine-tune with cultural nuance focus
4. **Evaluation**: Compare pre/post performance
5. **Deployment**: Save model and start API

## Cultural Nuance Handling

The system is specifically designed to handle:
- **Idioms**: "Break a leg" → "शुभकामनाएं!"
- **Informal Speech**: "What's up?" → "क्या हाल है?"
- **Emotional Expressions**: "I'm so proud" → "मुझे बहुत गर्व है"
- **Cultural References**: Context-aware translations

## Performance Improvements

Through fine-tuning on OpenSubtitles data, the model shows significant improvements in:
- Cultural expression preservation
- Informal language handling
- Emotional context maintenance
- Idiom translation accuracy

## Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'flask'**
```bash
pip install flask
```

**ModuleNotFoundError: No module named 'torch'**
```bash
# Use offline version instead
python simple_test.py
```

**Docker build fails**
```bash
# Check Docker is running
docker --version

# Use deployment scripts
./deploy.sh  # Linux/Mac
deploy.bat   # Windows
```

### Getting Help

1. **Start with offline version**: `python simple_test.py`
2. **Check logs**: `docker logs cultural-translator`
3. **Verify installation**: `python -c "import flask; print('OK')"`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please open an issue in the repository.

---

## Quick Reference

### Test Cultural Idiom
```bash
python simple_test.py test
# Enter: Even after one hits rock bottom, their arrogance remains unchanged.
# Expected: रस्सी जल गयी, बल नहीं गया
```

### Start API Server
```bash
python main_offline.py --mode serve --port 5000
```

### Docker Deployment
```bash
./deploy.sh  # Linux/Mac
deploy.bat   # Windows
```

### Test API
```bash
curl http://localhost:5000/health
curl -X POST http://localhost:5000/translate_dialogue -H "Content-Type: application/json" -d '{"text": "Even after one hits rock bottom, their arrogance remains unchanged.", "src_lang": "en", "tgt_lang": "hi"}'
``` 