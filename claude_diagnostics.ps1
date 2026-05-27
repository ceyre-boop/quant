#!/usr/bin/env pwsh
# ============================================================
#  ALTA / COLIN — CLAUDE CODE DIAGNOSTIC SCRIPT
#  Run with: pwsh claude_diagnostics.ps1
#  Saves full report to: ~/claude_diag_report.txt
# ============================================================

$ReportPath = "$HOME/claude_diag_report.txt"
$Divider    = "=" * 70
$SubDiv     = "-" * 50

function Section($title) {
    $ts = Get-Date -Format "HH:mm:ss"
    $block = @"

$Divider
  [$ts]  $title
$Divider
"@
    Write-Host $block -ForegroundColor Cyan
    Add-Content $ReportPath $block
}

function Sub($title) {
    $block = "`n$SubDiv`n  $title`n$SubDiv"
    Write-Host $block -ForegroundColor Yellow
    Add-Content $ReportPath $block
}

function Out($text) {
    Write-Host $text
    Add-Content $ReportPath $text
}

function Capture($label, [scriptblock]$cmd) {
    Sub $label
    try {
        $result = & $cmd 2>&1
        if ($result) {
            $str = $result | Out-String
            Out $str
        } else {
            Out "  (no output)"
        }
    } catch {
        Out "  ERROR: $_"
    }
}

# ── Init ──────────────────────────────────────────────────────────────────────
Clear-Host
Set-Content $ReportPath "CLAUDE CODE DIAGNOSTIC REPORT — $(Get-Date)`n"
Write-Host "`n  CLAUDE CODE DIAGNOSTIC — saving to $ReportPath`n" -ForegroundColor Green

# ══════════════════════════════════════════════════════════════════════════════
Section "1. SYSTEM IDENTITY & OS"
# ══════════════════════════════════════════════════════════════════════════════

Capture "macOS Version" { sw_vers }
Capture "Hardware Model" { system_profiler SPHardwareDataType | Select-String -Pattern "Model|Chip|Memory|Cores" }
Capture "PowerShell Version" { $PSVersionTable }
Capture "Shell Environment" {
    @{
        TERM      = $env:TERM
        LANG      = $env:LANG
        SHELL     = $env:SHELL
        PATH      = $env:PATH -split ":" | Select-Object -First 20
        COLORTERM = $env:COLORTERM
    }.GetEnumerator() | Sort-Object Name | Format-Table -AutoSize
}
Capture "Locale / Encoding" { locale }
Capture "Terminal Emulator Detection" {
    @{
        TERM_PROGRAM         = $env:TERM_PROGRAM
        TERM_PROGRAM_VERSION = $env:TERM_PROGRAM_VERSION
        ITERM_SESSION_ID     = $env:ITERM_SESSION_ID
        VSCODE_INJECTION     = $env:VSCODE_INJECTION
        WT_SESSION           = $env:WT_SESSION
    }.GetEnumerator() | Format-Table -AutoSize
}

# ══════════════════════════════════════════════════════════════════════════════
Section "2. CLAUDE CODE INSTALLATION"
# ══════════════════════════════════════════════════════════════════════════════

Capture "Claude Binary Location" { which claude }
Capture "Claude Version" { claude --version 2>&1 }
Capture "Claude Config List" { claude config list 2>&1 }
Capture "Claude Config File (~/.claude.json)" {
    $f = "$HOME/.claude.json"
    if (Test-Path $f) { Get-Content $f } else { "File not found: $f" }
}
Capture "Claude Config Directory Contents" {
    $d = "$HOME/.claude"
    if (Test-Path $d) { Get-ChildItem $d -Recurse | Select-Object FullName, Length, LastWriteTime } else { "Dir not found: $d" }
}

# ══════════════════════════════════════════════════════════════════════════════
Section "3. MCP SERVER INVENTORY"
# ══════════════════════════════════════════════════════════════════════════════

Capture "MCP Servers from ~/.claude.json (parsed)" {
    $f = "$HOME/.claude.json"
    if (Test-Path $f) {
        try {
            $json = Get-Content $f | ConvertFrom-Json
            if ($json.mcpServers) {
                $json.mcpServers | ConvertTo-Json -Depth 5
            } elseif ($json.mcp) {
                $json.mcp | ConvertTo-Json -Depth 5
            } else {
                "No mcpServers key found. Raw keys: $($json.PSObject.Properties.Name -join ', ')"
            }
        } catch {
            "JSON parse error: $_"
        }
    } else {
        "~/.claude.json not found"
    }
}

Capture "Project-level MCP configs (.mcp.json in common dirs)" {
    $searchPaths = @(
        "$HOME",
        "$HOME/Documents",
        "$HOME/Desktop",
        "$HOME/quant-1",
        "$HOME/Projects",
        "$HOME/Dev"
    )
    foreach ($p in $searchPaths) {
        if (Test-Path $p) {
            $found = Get-ChildItem $p -Filter ".mcp.json" -Depth 3 -ErrorAction SilentlyContinue
            if ($found) { $found | Select-Object FullName }
        }
    }
}

Capture "MCP-related Node processes currently running" {
    ps aux | grep -E "mcp|@anthropic|claude" | grep -v grep
}

# ══════════════════════════════════════════════════════════════════════════════
Section "4. NODE.JS & RUNTIME ENVIRONMENT"
# ══════════════════════════════════════════════════════════════════════════════

Capture "Node Version" { node --version 2>&1 }
Capture "NPM Version" { npm --version 2>&1 }
Capture "NPX Available" { which npx 2>&1 }
Capture "Global NPM Packages (Claude-related)" {
    npm list -g --depth=0 2>&1 | Select-String -Pattern "claude|mcp|anthropic|obsidian|agent|orchestra"
}
Capture "All Global NPM Packages" {
    npm list -g --depth=0 2>&1
}
Capture "NVM / Node Version Manager" {
    @(
        (which nvm 2>&1),
        (nvm list 2>&1),
        (which fnm 2>&1),
        (fnm list 2>&1)
    ) | Where-Object { $_ }
}

# ══════════════════════════════════════════════════════════════════════════════
Section "5. ACTIVE PROCESSES — CLAUDE / MCP / AGENTS"
# ══════════════════════════════════════════════════════════════════════════════

Capture "All Claude-related Processes" {
    ps aux | grep -iE "claude|anthropic|mcp" | grep -v grep
}
Capture "All Node Processes" {
    ps aux | grep -i node | grep -v grep
}
Capture "Python Processes (potential MCP servers)" {
    ps aux | grep -i python | grep -v grep
}
Capture "Obsidian Process" {
    ps aux | grep -i obsidian | grep -v grep
}
Capture "Port Listeners (potential MCP socket servers)" {
    netstat -an 2>/dev/null | grep LISTEN | grep -E ":(3[0-9]{3}|808[0-9]|8080|8181|9[0-9]{3})" | head -40
}
Capture "lsof Claude Sockets" {
    lsof -i -P | grep -iE "claude|mcp|3[0-9]{3}" | head -40
}

# ══════════════════════════════════════════════════════════════════════════════
Section "6. TERMINAL & ANSI CORRUPTION DIAGNOSTICS"
# ══════════════════════════════════════════════════════════════════════════════

Sub "Terminal capability check"
$termTests = @{
    "TERM value"       = $env:TERM
    "TERM_PROGRAM"     = $env:TERM_PROGRAM
    "COLORTERM"        = $env:COLORTERM
    "tput colors"      = (tput colors 2>&1)
    "tput cols"        = (tput cols 2>&1)
    "tput lines"       = (tput lines 2>&1)
    "stty size"        = (stty size 2>&1)
}
foreach ($k in $termTests.Keys) {
    $line = "  $k : $($termTests[$k])"
    Out $line
}

Capture "ANSI Reset Test (should show clean text)" {
    # Write ANSI reset then test string — if terminal is corrupt this will show garbage
    $ESC = [char]27
    "${ESC}[0m${ESC}[?1049l"
    Write-Host "  ANSI Reset Fired — if you see this cleanly, terminal buffer is healthy"
}

Capture "Locale Encoding (UTF-8 check)" {
    $enc = [System.Console]::OutputEncoding
    "Console Output Encoding : $($enc.EncodingName) (CodePage: $($enc.CodePage))"
    $enc2 = [System.Console]::InputEncoding
    "Console Input Encoding  : $($enc2.EncodingName) (CodePage: $($enc2.CodePage))"
}

# ══════════════════════════════════════════════════════════════════════════════
Section "7. MEMORY & PERFORMANCE"
# ══════════════════════════════════════════════════════════════════════════════

Capture "System Memory Pressure" {
    vm_stat | head -20
}
Capture "Top CPU/Mem Consumers" {
    ps aux --sort=-%mem 2>/dev/null | head -20
}
Capture "Disk Space" {
    df -h | grep -v tmpfs
}
Capture "Open File Descriptors (system)" {
    sysctl kern.maxfiles kern.maxfilesperproc 2>&1
}
Capture "Current Process FD Usage" {
    lsof | wc -l
}

# ══════════════════════════════════════════════════════════════════════════════
Section "8. OBSIDIAN / MEMORY TOOL DIAGNOSTICS"
# ══════════════════════════════════════════════════════════════════════════════

Capture "Obsidian Vaults (common locations)" {
    $vaultPaths = @(
        "$HOME/Library/Application Support/obsidian",
        "$HOME/Documents/Obsidian",
        "$HOME/Obsidian"
    )
    foreach ($p in $vaultPaths) {
        if (Test-Path $p) {
            Out "  FOUND: $p"
            Get-ChildItem $p -Depth 1 | Select-Object Name, LastWriteTime
        }
    }
}
Capture "Obsidian MCP Plugin Check" {
    $obsBase = "$HOME/Library/Application Support/obsidian"
    if (Test-Path $obsBase) {
        Get-ChildItem $obsBase -Recurse -Filter "*.json" -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match "mcp|claude|plugin" } |
            Select-Object FullName
    }
}

# ══════════════════════════════════════════════════════════════════════════════
Section "9. CLAUDE CODE LOGS"
# ══════════════════════════════════════════════════════════════════════════════

Capture "Claude Log Directory" {
    $logPaths = @(
        "$HOME/.claude/logs",
        "$HOME/Library/Logs/Claude",
        "/tmp/claude*"
    )
    foreach ($p in $logPaths) {
        if (Test-Path $p) {
            Out "  FOUND: $p"
            Get-ChildItem $p -Recurse -ErrorAction SilentlyContinue |
                Sort-Object LastWriteTime -Descending |
                Select-Object -First 10 FullName, Length, LastWriteTime
        }
    }
}

Capture "Most Recent Claude Log Tail (last 80 lines)" {
    $logDir = "$HOME/.claude/logs"
    if (Test-Path $logDir) {
        $latest = Get-ChildItem $logDir -File | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
            Out "  Reading: $($latest.FullName)"
            Get-Content $latest.FullName -Tail 80
        }
    } else {
        "No log dir found at $logDir"
    }
}

Capture "Crash Reports for Claude" {
    $crashDir = "$HOME/Library/Logs/DiagnosticReports"
    if (Test-Path $crashDir) {
        Get-ChildItem $crashDir | Where-Object { $_.Name -match "claude|node" } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 5 Name, LastWriteTime
    }
}

# ══════════════════════════════════════════════════════════════════════════════
Section "10. CONTEXT WINDOW / TOKEN BLOAT INDICATORS"
# ══════════════════════════════════════════════════════════════════════════════

Sub "Checking for known context-bloating patterns in MCP configs"
try {
    $f = "$HOME/.claude.json"
    if (Test-Path $f) {
        $raw = Get-Content $f -Raw
        $bloatPatterns = @("memory", "obsidian", "history", "inject", "context", "retrieval", "embed", "search")
        foreach ($pat in $bloatPatterns) {
            $matches = [regex]::Matches($raw, $pat, [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
            if ($matches.Count -gt 0) {
                Out "  [!] Pattern '$pat' found $($matches.Count) time(s) in config — potential context injector"
            }
        }
    }
} catch {
    Out "  Error scanning config: $_"
}

Capture "MCP Server Command Lines (what they actually execute)" {
    $f = "$HOME/.claude.json"
    if (Test-Path $f) {
        try {
            $json = Get-Content $f | ConvertFrom-Json
            $servers = $json.mcpServers
            if ($servers) {
                $servers.PSObject.Properties | ForEach-Object {
                    $name = $_.Name
                    $val  = $_.Value
                    Out "  SERVER: $name"
                    Out "    cmd  : $($val.command)"
                    Out "    args : $($val.args -join ' ')"
                    Out "    env  : $($val.env | ConvertTo-Json -Compress)"
                    Out ""
                }
            }
        } catch { Out "  Parse error: $_" }
    }
}

# ══════════════════════════════════════════════════════════════════════════════
Section "11. GIT REPOS CONNECTED TO CLAUDE CODE"
# ══════════════════════════════════════════════════════════════════════════════

Capture "Git Repos in Common Dev Locations" {
    $searchRoots = @("$HOME", "$HOME/Documents", "$HOME/Desktop", "$HOME/Dev", "$HOME/Projects")
    foreach ($root in $searchRoots) {
        if (Test-Path $root) {
            Get-ChildItem $root -Filter ".git" -Recurse -Depth 3 -Force -ErrorAction SilentlyContinue |
                Select-Object { Split-Path $_.FullName -Parent }
        }
    }
}

Capture "CLAUDE.md Files Found (context injection files)" {
    $searchRoots = @("$HOME", "$HOME/Documents", "$HOME/Desktop")
    foreach ($root in $searchRoots) {
        if (Test-Path $root) {
            Get-ChildItem $root -Filter "CLAUDE.md" -Recurse -Depth 4 -Force -ErrorAction SilentlyContinue |
                Select-Object FullName, Length, LastWriteTime
        }
    }
}

Capture "CLAUDE.md Contents (all found — these inject into every session)" {
    $searchRoots = @("$HOME", "$HOME/Documents", "$HOME/Desktop")
    foreach ($root in $searchRoots) {
        if (Test-Path $root) {
            $files = Get-ChildItem $root -Filter "CLAUDE.md" -Recurse -Depth 4 -Force -ErrorAction SilentlyContinue
            foreach ($f in $files) {
                Out "`n  ===== $($f.FullName) ====="
                Get-Content $f.FullName
            }
        }
    }
}

# ══════════════════════════════════════════════════════════════════════════════
Section "12. NETWORK — MCP SERVER REACHABILITY"
# ══════════════════════════════════════════════════════════════════════════════

Capture "Active Connections (potential MCP socket traffic)" {
    lsof -i TCP -P | grep -iE "ESTABLISHED|LISTEN" | head -30
}

# ══════════════════════════════════════════════════════════════════════════════
Section "13. SUMMARY FLAGS"
# ══════════════════════════════════════════════════════════════════════════════

Sub "Auto-generated risk flags"

$flags = @()

# Check MCP server count
try {
    $f = "$HOME/.claude.json"
    if (Test-Path $f) {
        $json = Get-Content $f | ConvertFrom-Json
        if ($json.mcpServers) {
            $count = ($json.mcpServers.PSObject.Properties | Measure-Object).Count
            if ($count -gt 5) { $flags += "[HIGH]  $count MCP servers registered — high context/process load risk" }
            elseif ($count -gt 2) { $flags += "[MED]   $count MCP servers — moderate load" }
            else { $flags += "[OK]    $count MCP servers registered" }
        }
    }
} catch {}

# Check Node version compatibility
try {
    $nv = (node --version 2>&1) -replace "v",""
    $major = [int]($nv -split "\.")[0]
    if ($major -lt 18) { $flags += "[HIGH]  Node.js v$nv is below v18 — Claude Code requires 18+" }
    else { $flags += "[OK]    Node.js v$nv" }
} catch { $flags += "[WARN]  Could not detect Node.js version" }

# Check TERM
if (-not $env:TERM -or $env:TERM -eq "dumb") {
    $flags += "[HIGH]  TERM is '$($env:TERM)' — dumb/missing TERM causes ANSI corruption"
} else {
    $flags += "[OK]    TERM=$($env:TERM)"
}

# Check LANG encoding
if ($env:LANG -notmatch "UTF-8") {
    $flags += "[MED]   LANG=$($env:LANG) — not UTF-8, may cause character corruption"
} else {
    $flags += "[OK]    LANG=$($env:LANG)"
}

foreach ($f in $flags) {
    $color = if ($f -match "\[HIGH\]") { "Red" } elseif ($f -match "\[MED\]") { "Yellow" } else { "Green" }
    Write-Host "  $f" -ForegroundColor $color
    Add-Content $ReportPath "  $f"
}

# ══════════════════════════════════════════════════════════════════════════════
$footer = @"

$Divider
  DIAGNOSTIC COMPLETE
  Report saved to: $ReportPath
  
  NEXT STEP: Copy the contents of $ReportPath
  and paste them into your Claude chat for analysis.
$Divider
"@
Write-Host $footer -ForegroundColor Green
Add-Content $ReportPath $footer