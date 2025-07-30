# 🚀 Railway Docker Deployment Guide

## Step-by-Step Instructions

### 1. Prepare Your Repository
- Make sure your code is pushed to GitHub
- Ensure you have these files in your repository:
  - `Dockerfile.offline`
  - `docker-compose.offline.yml`
  - `main_offline.py`
  - `requirements_offline.txt`

### 2. Deploy to Railway

1. **Go to [railway.app](https://railway.app)**
2. **Sign up/Login** with your GitHub account
3. **Click "New Project"**
4. **Select "Deploy from GitHub repo"**
5. **Choose your repository**
6. **Railway will automatically detect your Docker setup**

### 3. Configure Deployment

**Railway will automatically:**
- Detect your `Dockerfile.offline`
- Build your Docker image
- Deploy your container
- Give you a public URL

**If needed, manually configure:**
- **Service Name:** `cultural-translation-api`
- **Port:** `5000`
- **Environment Variables:** (usually not needed)

### 4. Get Your Production URL

Once deployed, Railway will give you:
- **Production URL:** `https://your-app-name.railway.app`
- **Custom Domain:** You can add your own domain

### 5. Test Your Production API

```bash
curl -X POST https://your-app-name.railway.app/translate_dialogue \
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

## Advantages of Railway
- ✅ **Automatic Docker detection**
- ✅ **Free tier available**
- ✅ **Instant public URL**
- ✅ **Auto-deploy from GitHub**
- ✅ **Custom domains**
- ✅ **SSL certificates included** 