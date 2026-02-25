#!/usr/bin/env bash
# smoke.sh — End-to-end smoke test for the Racing Sheets API
# Usage: ./scripts/smoke.sh [SESSION_ID]
# Requires: curl, jq
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
SESSION_ID="${1:-}"
PASS=0
FAIL=0

pass() { echo "  PASS: $1"; PASS=$((PASS + 1)); }
fail() { echo "  FAIL: $1"; FAIL=$((FAIL + 1)); }

echo "=== Smoke Test ==="
echo "API: $API_URL"
echo ""

# ------------------------------------------------------------------
# 1. Health check
# ------------------------------------------------------------------
echo "[1/5] Health check"
HEALTH=$(curl -sf "$API_URL/health" 2>/dev/null || echo "")
if [ -z "$HEALTH" ]; then
    fail "GET /health — no response (is the API running?)"
    echo ""
    echo "=== RESULT: $PASS passed, $FAIL failed ==="
    exit 1
fi
STATUS=$(echo "$HEALTH" | jq -r '.status // empty')
if [ "$STATUS" = "healthy" ]; then
    pass "GET /health -> status=healthy"
else
    fail "GET /health -> status=$STATUS (expected healthy)"
fi

# ------------------------------------------------------------------
# 2. DB stats
# ------------------------------------------------------------------
echo "[2/5] DB stats"
DB_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/db/stats" 2>/dev/null || echo "000")
if [ "$DB_CODE" = "200" ]; then
    pass "GET /db/stats -> 200"
else
    fail "GET /db/stats -> $DB_CODE (expected 200)"
fi

# ------------------------------------------------------------------
# 3. Resolve session
# ------------------------------------------------------------------
echo "[3/5] Session resolution"
if [ -z "$SESSION_ID" ]; then
    SESSIONS=$(curl -sf "$API_URL/sessions" 2>/dev/null || echo '{"sessions":[]}')
    SESSION_ID=$(echo "$SESSIONS" | jq -r '
        [.sessions[] | select(.has_primary == true)]
        | sort_by(.created_at) | reverse
        | .[0].session_id // empty')
fi

if [ -z "$SESSION_ID" ]; then
    echo "  SKIP: No sessions with primary data found — skipping bet build + odds checks"
    echo ""
    echo "=== RESULT: $PASS passed, $FAIL failed, 3 skipped ==="
    exit 0
else
    pass "Session: $SESSION_ID"
fi

# ------------------------------------------------------------------
# 4. Build bet plan
# ------------------------------------------------------------------
echo "[4/5] Bet plan build"
# Get session details for track/date
SESS_DETAIL=$(curl -sf "$API_URL/sessions" 2>/dev/null || echo '{"sessions":[]}')
TRACK=$(echo "$SESS_DETAIL" | jq -r ".sessions[] | select(.session_id==\"$SESSION_ID\") | .track // \"\"")
DATE=$(echo "$SESS_DETAIL" | jq -r ".sessions[] | select(.session_id==\"$SESSION_ID\") | .date // \"\"")

PLAN_HTTP=$(curl -s -o /tmp/smoke_plan.json -w "%{http_code}" -X POST "$API_URL/bets/build" \
    -H "Content-Type: application/json" \
    -d "{
        \"session_id\": \"$SESSION_ID\",
        \"track\": \"$TRACK\",
        \"race_date\": \"$DATE\",
        \"bankroll\": 1000,
        \"paper_mode\": true,
        \"allow_missing_odds\": true,
        \"save\": false
    }" 2>/dev/null || echo "000")
PLAN_RESP=$(cat /tmp/smoke_plan.json 2>/dev/null || echo "")

if [ "$PLAN_HTTP" = "404" ]; then
    echo "  SKIP: No predictions saved for this session (run projections first)"
elif [ -z "$PLAN_RESP" ] || [ "$PLAN_HTTP" = "000" ]; then
    fail "POST /bets/build — no response (HTTP $PLAN_HTTP)"
else
    HAS_DIAG=$(echo "$PLAN_RESP" | jq 'has("plan") and (.plan | has("diagnostics"))' 2>/dev/null || echo "false")
    if [ "$HAS_DIAG" = "true" ]; then
        pass "POST /bets/build -> plan.diagnostics present"
    else
        fail "POST /bets/build -> missing plan.diagnostics"
    fi

    TOTAL_TICKETS=$(echo "$PLAN_RESP" | jq '.plan.diagnostics.total_tickets // 0' 2>/dev/null || echo "0")
    BLOCKER_COUNT=$(echo "$PLAN_RESP" | jq '.plan.diagnostics.blockers | length // 0' 2>/dev/null || echo "0")
    if [ "$TOTAL_TICKETS" -gt 0 ] || [ "$BLOCKER_COUNT" -gt 0 ]; then
        pass "Tickets=$TOTAL_TICKETS, Blockers=$BLOCKER_COUNT (at least one non-zero)"
    else
        fail "Neither tickets nor blockers found in plan"
    fi
fi

# ------------------------------------------------------------------
# 5. Odds in tickets (if snapshots exist)
# ------------------------------------------------------------------
echo "[5/5] Odds verification"
ODDS_RESP=$(curl -sf "$API_URL/odds/snapshots/$SESSION_ID" 2>/dev/null || echo '{"snapshots":[]}')
SNAP_COUNT=$(echo "$ODDS_RESP" | jq '.snapshots | length' 2>/dev/null || echo "0")

if [ "$SNAP_COUNT" -eq 0 ]; then
    echo "  SKIP: No odds snapshots for this session"
else
    # Check if any ticket has odds_raw or odds in details
    if [ -n "$PLAN_RESP" ] && [ "$TOTAL_TICKETS" -gt 0 ]; then
        HAS_ODDS=$(echo "$PLAN_RESP" | jq '
            [.plan.race_plans[].tickets[]
             | select(.details.odds_raw != "" or .details.odds != null)]
            | length > 0' 2>/dev/null || echo "false")
        if [ "$HAS_ODDS" = "true" ]; then
            pass "Tickets contain odds fields (snapshots=$SNAP_COUNT)"
        else
            fail "Snapshots exist ($SNAP_COUNT) but no ticket has odds fields"
        fi
    else
        echo "  SKIP: No tickets to check odds against"
    fi
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "=== RESULT: $PASS passed, $FAIL failed ==="
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
