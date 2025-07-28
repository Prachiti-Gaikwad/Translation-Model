#!/bin/bash

# Cultural Translation Model Docker Deployment Script

echo "=========================================="
echo "Cultural Translation Model Deployment"
echo "=========================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Build and start the application
echo "🚀 Building and starting the application..."

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose.offline.yml down

# Build the image
echo "🔨 Building Docker image..."
docker-compose -f docker-compose.offline.yml build

# Start the services
echo "▶️ Starting services..."
docker-compose -f docker-compose.offline.yml up -d

# Wait for the service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 10

# Check if the service is running
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Service is running successfully!"
    echo ""
    echo "🌐 API Endpoints:"
    echo "  Health Check: http://localhost:5000/health"
    echo "  API Docs: http://localhost:5000/docs"
    echo "  Translation: http://localhost:5000/translate_dialogue"
    echo ""
    echo "📝 Test the cultural idiom:"
    echo "curl -X POST http://localhost:5000/translate_dialogue \\"
    echo "  -H \"Content-Type: application/json\" \\"
    echo "  -d '{\"text\": \"Even after one hits rock bottom, their arrogance remains unchanged.\", \"src_lang\": \"en\", \"tgt_lang\": \"hi\"}'"
    echo ""
    echo "🎬 Cultural Translation System is ready!"
else
    echo "❌ Service failed to start. Check logs with:"
    echo "docker-compose -f docker-compose.offline.yml logs"
fi 