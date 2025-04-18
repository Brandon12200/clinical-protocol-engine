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

# Set the root directory for Python imports
export PYTHONPATH=$(pwd)

# Determine the OS for the open browser command
OPEN_CMD=""
case "$(uname)" in
  "Darwin") # macOS
    OPEN_CMD="open"
    ;;
  "Linux")
    # Check for common browsers
    if command -v xdg-open > /dev/null; then
      OPEN_CMD="xdg-open"
    elif command -v gnome-open > /dev/null; then
      OPEN_CMD="gnome-open"
    fi
    ;;
  "MINGW"*|"MSYS"*|"CYGWIN"*) # Windows
    OPEN_CMD="start"
    ;;
esac

# Get the port from docker-compose.dev.yml
PORT=$(grep -o "[0-9]\+:5000" docker-compose.dev.yml | cut -d':' -f1)
if [ -z "$PORT" ]; then
  PORT=8080 # Default port if not found
fi

# Start development containers in the background
echo "Starting development environment..."

# Try docker-compose first, if it fails try docker compose (without hyphen)
if command -v docker-compose &> /dev/null; then
  docker-compose -f docker-compose.dev.yml up -d
else
  docker compose -f docker-compose.dev.yml up -d
fi

# Wait a moment for the containers to start
echo "Waiting for application to start..."
sleep 5

# Open the browser
echo "Opening http://localhost:$PORT in your browser..."
if [ -n "$OPEN_CMD" ]; then
  $OPEN_CMD "http://localhost:$PORT"
else
  echo "Could not detect browser command. Please open http://localhost:$PORT manually."
fi

# Stream the logs
echo "Streaming container logs (press Ctrl+C to stop viewing logs)..."
if command -v docker-compose &> /dev/null; then
  docker-compose -f docker-compose.dev.yml logs -f
else
  docker compose -f docker-compose.dev.yml logs -f
fi

# Note: Pressing Ctrl+C will only stop the log streaming, not the containers
echo "To stop the containers, run: docker compose -f docker-compose.dev.yml down"