# 🐳 Render Docker Deployment Guide

## Step-by-Step Instructions

### 1. Prepare Your Repository
- Make sure your code is pushed to GitHub
- Ensure you have these files:
  - `Dockerfile.offline`
  - `main_offline.py`
  - `requirements_offline.txt`

### 2. Deploy to Render

1. **Go to [render.com](https://render.com)**
2. **Sign up/Login** with your GitHub account
3. **Click "New +" → "Web Service"**
4. **Connect your GitHub repository**
5. **Configure the service:**

   **Basic Settings:**
   - **Name:** `cultural-translation-api`
   - **Environment:** `Docker`
   - **Region:** Choose closest to you
   - **Branch:** `main`

   **Build & Deploy:**
   - **Dockerfile Path:** `Dockerfile.offline`
   - **Docker Command:** Leave empty (uses CMD from Dockerfile)

   **Advanced Settings:**
   - **Auto-Deploy:** Yes
   - **Health Check Path:** `/health`

6. **Click "Create Web Service"**

### 3. Wait for Deployment
- Render will build your Docker image
- Deploy your container
- This usually takes 3-5 minutes

### 4. Get Your Production URL
- Once deployed, you'll get: `https://your-app-name.onrender.com`
- This is your **production URL** for submission!

### 5. Test Your Production API
```bash
curl -X POST https://your-app-name.onrender.com/translate_dialogue \
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