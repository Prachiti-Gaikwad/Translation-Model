@echo off
REM Cultural Translation Model Docker Deployment Script for Windows

echo ==========================================
echo Cultural Translation Model Deployment
echo ==========================================

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed. Please install Docker Compose first.
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are available

REM Build and start the application
echo 🚀 Building and starting the application...

REM Stop any existing containers
echo 🛑 Stopping existing containers...
docker-compose -f docker-compose.offline.yml down

REM Build the image
echo 🔨 Building Docker image...
docker-compose -f docker-compose.offline.yml build

REM Start the services
echo ▶️ Starting services...
docker-compose -f docker-compose.offline.yml up -d

REM Wait for the service to be ready
echo ⏳ Waiting for service to be ready...
timeout /t 10 /nobreak >nul

REM Check if the service is running
curl -f http://localhost:5000/health >nul 2>&1
if errorlevel 1 (
    echo ❌ Service failed to start. Check logs with:
    echo docker-compose -f docker-compose.offline.yml logs
    pause
    exit /b 1
) else (
    echo ✅ Service is running successfully!
    echo.
    echo 🌐 API Endpoints:
    echo   Health Check: http://localhost:5000/health
    echo   API Docs: http://localhost:5000/docs
    echo   Translation: http://localhost:5000/translate_dialogue
    echo.
    echo 📝 Test the cultural idiom:
    echo curl -X POST http://localhost:5000/translate_dialogue -H "Content-Type: application/json" -d "{\"text\": \"Even after one hits rock bottom, their arrogance remains unchanged.\", \"src_lang\": \"en\", \"tgt_lang\": \"hi\"}"
    echo.
    echo 🎬 Cultural Translation System is ready!
)

pause 