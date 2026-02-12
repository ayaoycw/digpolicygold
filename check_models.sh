#!/bin/bash
# Check which model deployments are available
ENDPOINT="https://forbrowseuse-resource.cognitiveservices.azure.com"
KEY=$(grep AZURE_OPENAI_API_KEY /opt/browser-sdk/.env | cut -d= -f2)

for model in gpt-4o gpt-4o-mini o3 o4-mini gpt-4.1-mini gpt-4.1 gpt-4.1-nano; do
    code=$(curl -sS -o /tmp/model_test.json -w '%{http_code}' \
        "${ENDPOINT}/openai/deployments/${model}/chat/completions?api-version=2024-12-01-preview" \
        -H "api-key: ${KEY}" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":5}')
    if [ "$code" = "200" ] || [ "$code" = "429" ]; then
        echo "✅ ${model}: available (HTTP ${code})"
    else
        echo "❌ ${model}: not found (HTTP ${code})"
    fi
done
