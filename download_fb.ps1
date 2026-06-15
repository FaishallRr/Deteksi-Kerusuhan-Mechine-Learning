# Download Facebook Reels ke folder indonesia_v4
# Usage: ./download_fb.ps1 "URL1" "URL2" ...
# Contoh: ./download_fb.ps1 "https://web.facebook.com/share/r/1KoHeuyaQ4/" "https://web.facebook.com/share/r/1YM8cF2dxG/"

param (
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Urls
)

$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$outputDir = Join-Path $scriptPath "sample_videos\indonesia_v4"
$ffmpegPath = Join-Path $scriptPath "bin\ffmpeg.exe"
$metadataFile = Join-Path $outputDir "_metadata.json"

if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

if ($Urls.Count -eq 0) {
    Write-Host "ERROR: Kasih URL Facebook reels" -ForegroundColor Red
    Write-Host "Contoh: .\download_fb.ps1 https://web.facebook.com/share/r/1KoHeuyaQ4/" -ForegroundColor Yellow
    exit 1
}

Write-Host "===== DOWNLOAD FACEBOOK REELS =====" -ForegroundColor Cyan
Write-Host "Output : $outputDir"
Write-Host "FFmpeg : $ffmpegPath"
Write-Host "Jumlah : $($Urls.Count) video"
Write-Host "====================================" -ForegroundColor Cyan

# Load existing metadata if any
$existing = @{}
if (Test-Path $metadataFile) {
    try { $existing = Get-Content $metadataFile | ConvertFrom-Json -AsHashtable } catch {}
}

$total = $Urls.Count
for ($i = 0; $i -lt $total; $i++) {
    $url = $Urls[$i].Trim()
    $pct = [math]::Round(($i / $total) * 100)
    Write-Host "[$($i+1)/$total - $pct%] Downloading: $url" -ForegroundColor Green
    
    $result = python -m yt_dlp `
        --ffmpeg-location $ffmpegPath `
        -o "$outputDir\%(id)s.%(ext)s" `
        --no-playlist `
        --no-overwrites `
        --cookies-from-browser chrome `
        --quiet `
        $url 2>&1

    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK" -ForegroundColor Green
    } else {
        Write-Host "  FAILED, coba tanpa cookies..." -ForegroundColor Yellow
        $result2 = python -m yt_dlp `
            --ffmpeg-location $ffmpegPath `
            -o "$outputDir\%(id)s.%(ext)s" `
            --no-playlist `
            --no-overwrites `
            --quiet `
            $url 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  OK (public)" -ForegroundColor Green
        } else {
            Write-Host "  GAGAL: Login Facebook dulu di Chrome" -ForegroundColor Red
        }
    }

    # Simpan metadata (mapping URL -> file)
    try {
        $existing[$url] = @{
            downloaded_at = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
            status = if ($LASTEXITCODE -eq 0) { "success" } else { "failed" }
        }
        $existing | ConvertTo-Json | Set-Content $metadataFile
    } catch {}
}

# Summary
Write-Host "`n=== Selesai ===" -ForegroundColor Cyan
$files = Get-ChildItem $outputDir -Exclude "_metadata.json"
Write-Host "File di $outputDir : $($files.Count)" -ForegroundColor Cyan
$files | ForEach-Object { Write-Host "  $($_.Name) ($([math]::Round($_.Length/1MB, 2)) MB)" }
