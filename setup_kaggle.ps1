# Setup Kaggle API credentials (Windows)
$kaggleDir = "$env:USERPROFILE\.kaggle"
$token = "KGAT_39d3c62a5068bc25f79172b4084c0689"

# Create .kaggle directory
mkdir $kaggleDir -Force | Out-Null

# Save access token
Set-Content -Path "$kaggleDir\access_token" -Value $token

Write-Host "✅ Kaggle API configured successfully"
Write-Host "📍 Token saved to: $kaggleDir\access_token"

# Test connection
Write-Host "`n🔄 Testing connection..."
python -m kaggle competitions list --max-results 3
