# smoke.ps1 — End-to-end smoke test for the Racing Sheets API
# Usage: .\scripts\smoke.ps1 [-SessionId <id>]
# Requires: PowerShell 5.1+
param(
    [string]$SessionId = "",
    [string]$ApiUrl = ""
)

$ErrorActionPreference = "Stop"
if (-not $ApiUrl) { $ApiUrl = if ($env:API_URL) { $env:API_URL } else { "http://localhost:8000" } }

$Pass = 0
$Fail = 0

function Write-Pass($msg) { Write-Host "  PASS: $msg" -ForegroundColor Green; $script:Pass++ }
function Write-Fail($msg) { Write-Host "  FAIL: $msg" -ForegroundColor Red; $script:Fail++ }

Write-Host "=== Smoke Test ==="
Write-Host "API: $ApiUrl"
Write-Host ""

# ------------------------------------------------------------------
# 1. Health check
# ------------------------------------------------------------------
Write-Host "[1/5] Health check"
try {
    $health = Invoke-RestMethod -Uri "$ApiUrl/health" -TimeoutSec 10
    if ($health.status -eq "healthy") {
        Write-Pass "GET /health -> status=healthy"
    } else {
        Write-Fail "GET /health -> status=$($health.status) (expected healthy)"
    }
} catch {
    Write-Fail "GET /health — no response (is the API running?)"
    Write-Host ""
    Write-Host "=== RESULT: $Pass passed, $Fail failed ==="
    exit 1
}

# ------------------------------------------------------------------
# 2. DB stats
# ------------------------------------------------------------------
Write-Host "[2/5] DB stats"
try {
    $null = Invoke-RestMethod -Uri "$ApiUrl/db/stats" -TimeoutSec 10
    Write-Pass "GET /db/stats -> 200"
} catch {
    Write-Fail "GET /db/stats -> $($_.Exception.Response.StatusCode)"
}

# ------------------------------------------------------------------
# 3. Resolve session
# ------------------------------------------------------------------
Write-Host "[3/5] Session resolution"
if (-not $SessionId) {
    try {
        $sessResp = Invoke-RestMethod -Uri "$ApiUrl/sessions" -TimeoutSec 10
        $primary = $sessResp.sessions |
            Where-Object { $_.has_primary -eq $true } |
            Sort-Object { $_.created_at } -Descending |
            Select-Object -First 1
        if ($primary) { $SessionId = $primary.session_id }
    } catch { }
}

if (-not $SessionId) {
    Write-Host "  SKIP: No sessions with primary data found"
    Write-Host ""
    Write-Host "=== RESULT: $Pass passed, $Fail failed, 3 skipped ==="
    exit 0
} else {
    Write-Pass "Session: $SessionId"
}

# ------------------------------------------------------------------
# 4. Build bet plan
# ------------------------------------------------------------------
Write-Host "[4/5] Bet plan build"
$Track = ""
$Date = ""
try {
    $sessResp = Invoke-RestMethod -Uri "$ApiUrl/sessions" -TimeoutSec 10
    $match = $sessResp.sessions | Where-Object { $_.session_id -eq $SessionId } | Select-Object -First 1
    if ($match) {
        $Track = if ($match.track) { $match.track } else { "" }
        $Date = if ($match.date) { $match.date } else { "" }
    }
} catch { }

$PlanResp = $null
$TotalTickets = 0
try {
    $body = @{
        session_id = $SessionId
        track = $Track
        race_date = $Date
        bankroll = 1000
        paper_mode = $true
        allow_missing_odds = $true
        save = $false
    } | ConvertTo-Json

    $PlanResp = Invoke-RestMethod -Uri "$ApiUrl/bets/build" -Method Post `
        -ContentType "application/json" -Body $body -TimeoutSec 30

    if ($PlanResp.plan -and $PlanResp.plan.diagnostics) {
        Write-Pass "POST /bets/build -> plan.diagnostics present"
    } else {
        Write-Fail "POST /bets/build -> missing plan.diagnostics"
    }

    $TotalTickets = if ($PlanResp.plan.diagnostics.total_tickets) {
        $PlanResp.plan.diagnostics.total_tickets
    } else { 0 }
    $blockers = if ($PlanResp.plan.diagnostics.blockers) {
        $PlanResp.plan.diagnostics.blockers
    } else { @() }

    if ($TotalTickets -gt 0 -or $blockers.Count -gt 0) {
        Write-Pass "Tickets=$TotalTickets, Blockers=$($blockers.Count)"
    } else {
        Write-Fail "Neither tickets nor blockers found"
    }
} catch {
    Write-Fail "POST /bets/build — $($_.Exception.Message)"
}

# ------------------------------------------------------------------
# 5. Odds in tickets
# ------------------------------------------------------------------
Write-Host "[5/5] Odds verification"
$SnapCount = 0
try {
    $oddsResp = Invoke-RestMethod -Uri "$ApiUrl/odds/snapshots/$SessionId" -TimeoutSec 10
    $SnapCount = if ($oddsResp.snapshots) { $oddsResp.snapshots.Count } else { 0 }
} catch { }

if ($SnapCount -eq 0) {
    Write-Host "  SKIP: No odds snapshots for this session"
} elseif ($PlanResp -and $TotalTickets -gt 0) {
    $hasOdds = $false
    foreach ($rp in $PlanResp.plan.race_plans) {
        foreach ($t in $rp.tickets) {
            if ($t.details.odds_raw -or $t.details.odds) {
                $hasOdds = $true
                break
            }
        }
        if ($hasOdds) { break }
    }
    if ($hasOdds) {
        Write-Pass "Tickets contain odds fields (snapshots=$SnapCount)"
    } else {
        Write-Fail "Snapshots exist ($SnapCount) but no ticket has odds fields"
    }
} else {
    Write-Host "  SKIP: No tickets to check odds against"
}

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
Write-Host ""
Write-Host "=== RESULT: $Pass passed, $Fail failed ==="
if ($Fail -gt 0) { exit 1 }
exit 0
