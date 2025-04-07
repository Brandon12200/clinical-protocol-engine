#!/bin/bash

# Load environment variables if .env file exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
  echo "Loaded environment variables from .env file"
else
  echo "No .env file found, using default environment variables"
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Start development containers
echo "Starting development environment..."

# Try docker-compose first, if it fails try docker compose (without hyphen)
if command -v docker-compose &> /dev/null; then
  docker-compose -f docker-compose.dev.yml up --build
else
  docker compose -f docker-compose.dev.yml up --build
fi

# Note: Use Ctrl+C to stop the containers