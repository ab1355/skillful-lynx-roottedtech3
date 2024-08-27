#!/bin/bash

# Ensure you're authenticated with Fly.io
# Run 'flyctl auth login' if not already authenticated

# Deploy the application
flyctl deploy --config fly.staging.toml

# Output the staging URL
echo "Staging environment is ready at: https://system32-long-surf-1343.fly.dev"

# Scale the app to ensure at least one instance is running
flyctl scale count 1 --app system32-long-surf-1343

# Check the status of the deployment
flyctl status --app system32-long-surf-1343