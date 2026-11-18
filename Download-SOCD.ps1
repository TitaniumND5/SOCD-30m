[CmdletBinding()]
param(
  [ValidateSet('TIFF','OVR','All')]
  [string]$Set = 'TIFF',
  [string]$OutputDirectory = (Get-Location).Path
)
$ErrorActionPreference = 'Stop'
$Repository = 'TitaniumND5/SOCD-30m'
$Tag = 'data-v1'
$ApiUri = "https://api.github.com/repos/$Repository/releases/tags/$Tag"
$ManifestUri = "https://raw.githubusercontent.com/$Repository/main/SHA256SUMS.csv"
$Headers = @{
  Accept = 'application/vnd.github+json'
  'X-GitHub-Api-Version' = '2022-11-28'
  'User-Agent' = 'SOCD-30m-download-helper'
}
$Curl = Join-Path $env:SystemRoot 'System32\curl.exe'
if (-not (Test-Path -LiteralPath $Curl)) { $Curl = (Get-Command curl.exe -ErrorAction Stop).Source }
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$Release = Invoke-RestMethod -Uri $ApiUri -Headers $Headers -Method Get
if ($Release.draft) { throw 'Release is not published yet.' }
$ManifestResponse = Invoke-WebRequest -Uri $ManifestUri -Headers $Headers -UseBasicParsing
$ManifestRows = @(ConvertFrom-Csv -InputObject $ManifestResponse.Content)
$Wanted = @()
if ($Set -eq 'TIFF' -or $Set -eq 'All') { $Wanted += 'SOCD_30m_TIFF' }
if ($Set -eq 'OVR' -or $Set -eq 'All') { $Wanted += 'SOCD_30m_OVR' }
$Assets = @($Release.assets | Where-Object {
  $name = $_.name
  foreach ($prefix in $Wanted) {
    if ($name -match ("^{0}\.7z\.\d{3}$" -f [regex]::Escape($prefix))) { return $true }
  }
  return $false
} | Sort-Object name)
if ($Assets.Count -eq 0) { throw "No release assets found for set $Set." }
foreach ($Asset in $Assets) {
  $ManifestRow = $ManifestRows | Where-Object { $_.Name -eq $Asset.name } | Select-Object -First 1
  $ExpectedHash = $null
  if ($Asset.digest -match '^sha256:([0-9a-fA-F]{64})$') { $ExpectedHash = $Matches[1].ToLowerInvariant() }
  elseif ($ManifestRow -and $ManifestRow.SHA256 -match '^[0-9a-fA-F]{64}$') { $ExpectedHash = $ManifestRow.SHA256.ToLowerInvariant() }
  if (-not $ExpectedHash) { throw "No usable SHA-256 is available for $($Asset.name)." }
  $Destination = Join-Path $OutputDirectory $Asset.name
  $Partial = "$Destination.part"
  if (Test-Path -LiteralPath $Destination) {
    $Existing = Get-Item -LiteralPath $Destination
    if ($Existing.Length -eq [long]$Asset.size -and (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $ExpectedHash) {
      Write-Host "Verified, already complete: $($Asset.name)"
      continue
    }
    throw "Existing file does not match the release; move it before retrying: $Destination"
  }
  if ((Test-Path -LiteralPath $Partial) -and (Get-Item -LiteralPath $Partial).Length -gt [long]$Asset.size) {
    throw "Partial file is larger than the release asset; inspect it before retrying: $Partial"
  }
  if ((Test-Path -LiteralPath $Partial) -and (Get-Item -LiteralPath $Partial).Length -eq [long]$Asset.size) {
    $PartHash = (Get-FileHash -LiteralPath $Partial -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($PartHash -eq $ExpectedHash) {
      Move-Item -LiteralPath $Partial -Destination $Destination
      Write-Host "Verified and finalized: $($Asset.name)"
      continue
    }
    $Quarantine = "$Partial.corrupt.$(Get-Date -Format yyyyMMddHHmmss)"
    Move-Item -LiteralPath $Partial -Destination $Quarantine
  }
  $Done = $false
  for ($Attempt = 1; $Attempt -le 4; $Attempt++) {
    Write-Host "Downloading $($Asset.name) (attempt $Attempt/4)"
    & $Curl --fail --location --show-error --retry 2 --retry-delay 3 --retry-connrefused --continue-at - --output $Partial $Asset.browser_download_url
    if ($LASTEXITCODE -eq 0) { $Done = $true; break }
    if ($Attempt -lt 4) {
      $Delay = [int][Math]::Min(120, 15 * [Math]::Pow(2, $Attempt - 1))
      Write-Warning "Transfer failed (curl exit $LASTEXITCODE); retaining partial data and retrying in $Delay seconds."
      Start-Sleep -Seconds $Delay
    }
  }
  if (-not $Done) { throw "Download failed after retries: $($Asset.name). Partial data is retained." }
  $Downloaded = Get-Item -LiteralPath $Partial
  if ($Downloaded.Length -ne [long]$Asset.size) { throw "Size mismatch for $($Asset.name): got $($Downloaded.Length), expected $($Asset.size)." }
  $ActualHash = (Get-FileHash -LiteralPath $Partial -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($ActualHash -ne $ExpectedHash) {
    $Quarantine = "$Partial.corrupt.$(Get-Date -Format yyyyMMddHHmmss)"
    Move-Item -LiteralPath $Partial -Destination $Quarantine
    throw "SHA-256 mismatch for $($Asset.name); bad partial retained as $Quarantine."
  }
  Move-Item -LiteralPath $Partial -Destination $Destination
  Write-Host "Verified: $($Asset.name) [$($Downloaded.Length) bytes]"
}
Write-Host "Completed and verified set: $Set"
