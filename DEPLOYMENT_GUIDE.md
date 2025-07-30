# 🚀 Cultural Translation API - Deployment Guide

## Overview
This guide will help you deploy the Cultural Translation API to production platforms for your technical assessment.

## 🎯 Quick Deployment Options

### Option 1: Render (Recommended - Free & Easy)
1. **Sign up** at [render.com](https://render.com)
2. **Connect your GitHub repository**
3. **Create a new Web Service**
4. **Configure:**
   - **Build Command:** `pip install -r requirements_offline.txt`
   - **Start Command:** `python main_offline.py --mode serve --port $PORT`
   - **Environment:** Python 3.9

### Option 2: Railway (Alternative - Free Tier)
1. **Sign up** at [railway.app](https://railway.app)
2. **Connect GitHub repository**
3. **Deploy automatically**

### Option 3: Heroku (Paid)
1. **Sign up** at [heroku.com](https://heroku.com)
2. **Install Heroku CLI**
3. **Deploy using:** `git push heroku main`

## 📁 Required Files for Deployment

Ensure these files are in your repository:
- ✅ `main_offline.py` - Main API application
- ✅ `requirements_offline.txt` - Python dependencies
- ✅ `render.yaml` - Render deployment config (optional)
- ✅ `README.md` - Project documentation

## 🔧 Local Testing Before Deployment

### Test the API Locally:
```bash
# Install dependencies
pip install -r requirements_offline.txt

# Run the API
python main_offline.py --mode serve --port 5000
```

### Test API Endpoints:
```bash
# Health check
curl http://localhost:5000/health

# Get API documentation
curl http://localhost:5000/docs

# Test translation (the task-specific example)
curl -X POST http://localhost:5000/translate_dialogue \
  -H "Content-Type: application/json" \
  -d '{"text": "Even after one hits rock bottom, their arrogance remains unchanged."}'
```

## 🌐 Production Deployment Steps

### Step 1: Prepare Repository
```bash
# Ensure all files are committed
git add .
git commit -m "Prepare for production deployment"
git push origin main
```

### Step 2: Deploy to Render
1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name:** `cultural-translation-api`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements_offline.txt`
   - **Start Command:** `python main_offline.py --mode serve --port $PORT`
5. Click "Create Web Service"

### Step 3: Get Production URL
- Render will provide a URL like: `https://your-app-name.onrender.com`
- This is your **production deployment link**

## 🧪 Testing Production Deployment

### Test the Production API:
```bash
# Replace with your actual production URL
PROD_URL="https://your-app-name.onrender.com"

# Health check
curl $PROD_URL/health

# API documentation
curl $PROD_URL/docs

# Test the task-specific cultural idiom
curl -X POST $PROD_URL/translate_dialogue \
  -H "Content-Type: application/json" \
  -d '{"text": "Even after one hits rock bottom, their arrogance remains unchanged."}'
```

### Expected Response:
```json
{
  "translation": "रस्सी जल गयी, बल नहीं गया",
  "original": "Even after one hits rock bottom, their arrogance remains unchanged.",
  "confidence": "high",
  "source_language": "en",
  "target_language": "hi"
}
```

## 📋 Submission Checklist

Before submitting to evaluators, ensure:

- ✅ **Production URL is working** (test all endpoints)
- ✅ **API responds correctly** to the task-specific example
- ✅ **Documentation is accessible** via `/docs` endpoint
- ✅ **Health check passes** via `/health` endpoint
- ✅ **Code is in GitHub repository** with proper documentation
- ✅ **README.md** contains clear instructions

## 🔍 Troubleshooting

### Common Issues:

1. **Build fails on Render:**
   - Check `requirements_offline.txt` has correct dependencies
   - Ensure Python version is compatible

2. **API not responding:**
   - Check logs in Render dashboard
   - Verify start command is correct

3. **Port issues:**
   - Render uses `$PORT` environment variable
   - Local testing uses port 5000

### Debug Commands:
```bash
# Check if API is running locally
curl http://localhost:5000/health

# Check Python dependencies
pip list | grep flask

# Test specific endpoint
python -c "
import requests
response = requests.post('http://localhost:5000/translate_dialogue', 
  json={'text': 'Even after one hits rock bottom, their arrogance remains unchanged.'})
print(response.json())
"
```

## 📞 Support

If you encounter issues:
1. Check Render logs in the dashboard
2. Test locally first
3. Verify all required files are present
4. Ensure GitHub repository is public (for free Render deployment)

## 🎉 Success!

Once deployed, you'll have:
- ✅ **Working production API**
- ✅ **Public URL for evaluators**
- ✅ **Complete documentation**
- ✅ **Task-specific cultural idiom support**

**Your production link will be:** `https://your-app-name.onrender.com` 