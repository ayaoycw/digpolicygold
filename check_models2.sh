#!/bin/bash
ENDPOINT="https://forbrowseuse-resource.cognitiveservices.azure.com"
KEY=$(grep AZURE_OPENAI_API_KEY /opt/browser-sdk/.env | cut -d= -f2)

for model in o3 o4-mini o3-mini; do
    code=$(curl -sS -o /tmp/model_test.json -w '%{http_code}' \
        "${ENDPOINT}/openai/deployments/${model}/chat/completions?api-version=2024-12-01-preview" \
        -H "api-key: ${KEY}" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"hi"}]}')
    echo "${model}: HTTP ${code}"
    cat /tmp/model_test.json | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  model:", d.get("model","?"))' 2>/dev/null
done
