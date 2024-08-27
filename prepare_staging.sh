#!/bin/bash

# Install Flyctl if not already installed
if ! command -v flyctl &> /dev/null
then
    curl -L https://fly.io/install.sh | sh
fi

# Deploy to staging
flyctl deploy --config fly.staging.toml

# Output the staging URL
echo "Staging environment is ready at: https://agentzero-staging.fly.dev"