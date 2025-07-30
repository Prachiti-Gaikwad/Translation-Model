# 🐳 Heroku Docker Deployment Guide

## Step-by-Step Instructions

### 1. Install Heroku CLI
```bash
# Download from: https://devcenter.heroku.com/articles/heroku-cli
# Or use winget on Windows:
winget install --id=Heroku.HerokuCLI
```

### 2. Login to Heroku
```bash
heroku login
```

### 3. Create Heroku App
```bash
# In your project directory
heroku create your-app-name
```

### 4. Deploy with Docker
```bash
# Heroku will automatically detect your Dockerfile
heroku container:push web
heroku container:release web
```

### 5. Open Your App
```bash
heroku open
```

### 6. Get Your Production URL
- Your app will be available at: `https://your-app-name.herokuapp.com`
- This is your **production URL** for submission!

### 7. Test Your Production API
```bash
curl -X POST https://your-app-name.herokuapp.com/translate_dialogue \
  -H "Content-Type: application/json" \
  -d '{"text": "Even after one hits rock bottom, their arrogance remains unchanged."}'
```

## Expected Response
```json
{
  "confidence": "high",
  "original": "Even after one hits rock bottom, their arrogance remains unchanged.",
  "source_language": "en",
  "target_language": "hi",
  "translation": "रस्सी जल गयी, बल नहीं गया"
}
``` 