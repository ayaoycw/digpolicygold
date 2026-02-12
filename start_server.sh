#!/bin/bash
# Start Policy Search API server
cd /opt/browser-sdk
source .venv/bin/activate

# Kill any existing server
pkill -f "python.*server.py" 2>/dev/null
sleep 1

# Start with xvfb for headless browser
echo "Starting server..."
nohup xvfb-run --auto-servernum python3 -u server.py > /tmp/server.log 2>&1 &
sleep 3

# Check if started
cat /tmp/server.log
echo ""
echo "Server PID: $(pgrep -f 'python.*server.py')"
