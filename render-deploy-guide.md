# 🚀 Render Deployment Guide

## Step-by-Step Instructions

### 1. Prepare Your Repository
- Make sure your code is pushed to GitHub
- Ensure `requirements_offline.txt` and `main_offline.py` are in the root directory

### 2. Deploy to Render

1. **Go to [render.com](https://render.com)**
2. **Sign up/Login** with your GitHub account
3. **Click "New +" → "Web Service"**
4. **Connect your GitHub repository**
5. **Configure the service:**

   **Basic Settings:**
   - **Name:** `cultural-translation-api`
   - **Environment:** `Python 3`
   - **Region:** Choose closest to you
   - **Branch:** `main` (or your default branch)

   **Build & Deploy:**
   - **Build Command:** `pip install -r requirements_offline.txt`
   - **Start Command:** `python main_offline.py --mode serve --port $PORT`

   **Advanced Settings:**
   - **Auto-Deploy:** Yes
   - **Health Check Path:** `/health`

6. **Click "Create Web Service"**

### 3. Wait for Deployment
- Render will automatically build and deploy your app
- This usually takes 2-5 minutes
- You'll see build logs in real-time

### 4. Get Your Production URL
- Once deployed, you'll get a URL like: `https://your-app-name.onrender.com`
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

## Troubleshooting
- If build fails, check the logs in Render dashboard
- Make sure all files are committed to GitHub
- Verify `requirements_offline.txt` contains: `flask==2.3.3` 