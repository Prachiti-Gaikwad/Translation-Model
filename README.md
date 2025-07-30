# Cultural Translation API

A simple Flask API for translating English text to Hindi with cultural nuance preservation.

## Features

- Translates English text to Hindi
- Preserves cultural idioms and expressions
- Simple REST API
- Ready for cloud deployment

## Quick Start

### Local Development
```bash
pip install flask==2.3.3
python app.py
```

### Test the API
```bash
curl -X POST http://localhost:5000/translate_dialogue \
  -H "Content-Type: application/json" \
  -d '{"text": "Even after one hits rock bottom, their arrogance remains unchanged."}'
```

## API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `POST /translate_dialogue` - Translate text

## Deployment

This app is ready for deployment on:
- Railway
- Render
- Heroku
- Any Python hosting platform

## Example Response

```json
{
  "translation": "रस्सी जल गयी, बल नहीं गया",
  "confidence": "high",
  "original": "Even after one hits rock bottom, their arrogance remains unchanged.",
  "source_language": "en",
  "target_language": "hi"
}
``` 