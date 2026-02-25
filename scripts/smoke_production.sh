#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-https://api.loopstar23.com}"
APP_URL="${APP_URL:-https://app.loopstar23.com}"

echo "=== Production Smoke Test ==="
echo "API: $API_URL"
echo "APP: $APP_URL"
echo ""

# 1. API health
echo "[1/5] API health check..."
STATUS=$(curl -sf "$API_URL/health" -o /dev/null -w "%{http_code}" || echo "FAIL")
if [ "$STATUS" = "200" ]; then
    echo "  PASS: /health returned 200"
else
    echo "  FAIL: /health returned $STATUS"
fi

# 2. API root
echo "[2/5] API root..."
STATUS=$(curl -sf "$API_URL/" -o /dev/null -w "%{http_code}" || echo "FAIL")
echo "  / returned HTTP $STATUS"

# 3. Streamlit responds
echo "[3/5] Streamlit frontend..."
STATUS=$(curl -sf "$APP_URL" -o /dev/null -w "%{http_code}" || echo "FAIL")
if [ "$STATUS" = "200" ]; then
    echo "  PASS: Streamlit returned 200"
else
    echo "  FAIL: Streamlit returned $STATUS"
fi

# 4. CORS check
echo "[4/5] CORS headers..."
CORS=$(curl -s -I -H "Origin: $APP_URL" "$API_URL/health" 2>/dev/null | grep -i "access-control-allow-origin" || echo "NOT FOUND")
echo "  $CORS"

# 5. Sessions endpoint (may require auth in Phase 2)
echo "[5/5] Sessions endpoint..."
STATUS=$(curl -sf "$API_URL/sessions" -o /dev/null -w "%{http_code}" 2>/dev/null || echo "FAIL")
echo "  /sessions returned HTTP $STATUS"

echo ""
echo "=== Smoke Test Complete ==="
