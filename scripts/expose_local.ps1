# =============================================================================
#  expose_local.ps1 — Expose tous les services via Cloudflare Quick Tunnels
#  Plan B si VPS indisponible — fonctionne depuis le PC local
#
#  Usage : powershell -ExecutionPolicy Bypass -File scripts\expose_local.ps1
# =============================================================================

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  MLOps Rakuten — Exposition publique (Plan B)" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# ─── 1. Installer cloudflared si absent ──────────────────────────────────────
if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) {
    Write-Host "[1/3] Installation de cloudflared..." -ForegroundColor Yellow
    winget install --id Cloudflare.cloudflared --silent --accept-source-agreements --accept-package-agreements
    # Recharger le PATH
    $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("PATH","User")
    Start-Sleep -Seconds 3
} else {
    Write-Host "[1/3] cloudflared deja installe" -ForegroundColor Green
}

# ─── 2. Demarrer Docker Compose ──────────────────────────────────────────────
Write-Host "[2/3] Demarrage de Docker Compose..." -ForegroundColor Yellow
Set-Location (Split-Path $PSScriptRoot -Parent)

# Verifier si les services tournent deja
$running = docker compose ps --services --filter "status=running" 2>$null
if ($running -match "streamlit") {
    Write-Host "      Services Docker deja actifs" -ForegroundColor Green
} else {
    Write-Host "      Lancement des services (peut prendre 2-3 min)..." -ForegroundColor Yellow
    docker compose up -d
    Write-Host "      Attente demarrage API (healthcheck)..." -ForegroundColor Yellow
    $timeout = 120
    $elapsed = 0
    do {
        Start-Sleep -Seconds 5
        $elapsed += 5
        $health = docker inspect --format='{{.State.Health.Status}}' (docker compose ps -q api 2>$null) 2>$null
    } while ($health -ne "healthy" -and $elapsed -lt $timeout)

    if ($health -eq "healthy") {
        Write-Host "      API OK" -ForegroundColor Green
    } else {
        Write-Host "      API pas encore healthy — on continue quand meme" -ForegroundColor Yellow
    }
}

# ─── 3. Lancer les tunnels Cloudflare ────────────────────────────────────────
Write-Host "[3/3] Creation des tunnels Cloudflare..." -ForegroundColor Yellow
Write-Host "      (les URLs apparaitront dans ~10 secondes)" -ForegroundColor Gray
Write-Host ""

$services = @(
    @{ Name = "Streamlit  "; Port = 8501 },
    @{ Name = "MLflow     "; Port = 5000 },
    @{ Name = "Airflow    "; Port = 8280 },
    @{ Name = "Grafana    "; Port = 3000 },
    @{ Name = "API Swagger"; Port = 8200 },
    @{ Name = "Prometheus "; Port = 9090 }
)

$jobs = @()
$logDir = "$env:TEMP\cf-tunnels"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($svc in $services) {
    $logFile = "$logDir\$($svc.Port).log"
    $job = Start-Job -ScriptBlock {
        param($port, $log)
        cloudflared tunnel --url "http://localhost:$port" 2>&1 | Tee-Object -FilePath $log
    } -ArgumentList $svc.Port, $logFile
    $jobs += @{ Job = $job; Service = $svc; Log = $logFile }
}

# Attendre que les URLs soient générées
Write-Host "Collecte des URLs publiques..." -ForegroundColor Yellow
Start-Sleep -Seconds 15

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  URLS PUBLIQUES — A partager avec le jury            " -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

$urlsFound = @()
foreach ($entry in $jobs) {
    $url = $null
    if (Test-Path $entry.Log) {
        $content = Get-Content $entry.Log -Raw -ErrorAction SilentlyContinue
        if ($content -match 'https://[a-z0-9\-]+\.trycloudflare\.com') {
            $url = $matches[0]
        }
    }
    if ($url) {
        Write-Host ("  {0} -> {1}" -f $entry.Service.Name, $url) -ForegroundColor Cyan
        $urlsFound += "  $($entry.Service.Name) -> $url"
    } else {
        Write-Host ("  {0} -> URL pas encore prête (vérifier {1})" -f $entry.Service.Name, $entry.Log) -ForegroundColor Red
    }
}

# Sauvegarder les URLs dans un fichier
$urlFile = ".\URLS_PUBLIQUES.txt"
@"
MLOps Rakuten — URLs publiques pour la soutenance
Générées le : $(Get-Date -Format 'dd/MM/yyyy HH:mm')
==============================================
$($urlsFound -join "`n")

Identifiants :
  Airflow  : airflow / airflow
  Grafana  : admin / admin
  API Token: rakuten-soutenance-2024
==============================================
IMPORTANT : Ces URLs changent à chaque redémarrage de cloudflared.
            Garder ce script actif pendant la soutenance.
"@ | Out-File -FilePath $urlFile -Encoding UTF8

Write-Host ""
Write-Host "  URLs sauvegardées dans : $urlFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "  CTRL+C pour arreter les tunnels                     " -ForegroundColor Red
Write-Host "======================================================" -ForegroundColor Green
Write-Host ""

# Garder le script actif
Write-Host "Tunnels actifs. Appuie sur CTRL+C pour arreter..." -ForegroundColor Yellow
try {
    while ($true) {
        Start-Sleep -Seconds 30
        # Verification que les jobs tournent encore
        foreach ($entry in $jobs) {
            if ($entry.Job.State -eq "Failed" -or $entry.Job.State -eq "Completed") {
                Write-Host "  AVERTISSEMENT: tunnel port $($entry.Service.Port) s'est arrete — relancement..." -ForegroundColor Red
                $logFile = "$logDir\$($entry.Service.Port).log"
                $newJob = Start-Job -ScriptBlock {
                    param($port, $log)
                    cloudflared tunnel --url "http://localhost:$port" 2>&1 | Tee-Object -FilePath $log -Append
                } -ArgumentList $entry.Service.Port, $logFile
                $entry.Job = $newJob
            }
        }
    }
} finally {
    Write-Host "Arret des tunnels..." -ForegroundColor Yellow
    $jobs | ForEach-Object { Stop-Job $_.Job; Remove-Job $_.Job }
}
