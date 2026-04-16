# -----------------------------------------------------------------------------
# update_template.ps1
#
# Updates the EC2 Launch Template with the latest user_data.sh contents.
#
# What it does:
# - Reads user_data.sh from the current folder
# - Encodes it to Base64
# - Injects it into ec2_gpu_launch_template.local.json
# - Creates a new Launch Template version in AWS
#
# Usage:
#   Run this script from infra/aws
#   .\update_template.ps1
# -----------------------------------------------------------------------------

$ErrorActionPreference = "Stop"

$LaunchTemplateName = "system2-gpu-template"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$UserDataPath = Join-Path $ScriptDir "user_data.sh"
$TemplateJsonPath = Join-Path $ScriptDir "ec2_gpu_launch_template.local.json"
$TempJsonPath = Join-Path $ScriptDir "_tmp_launch_template.json"

Write-Host "Reading user_data.sh..."
if (-not (Test-Path $UserDataPath)) {
    throw "user_data.sh not found at: $UserDataPath"
}
$encodedUserData = [Convert]::ToBase64String([IO.File]::ReadAllBytes($UserDataPath))

Write-Host "Preparing launch template JSON..."
if (-not (Test-Path $TemplateJsonPath)) {
    throw "Template JSON not found at: $TemplateJsonPath"
}

$json = Get-Content $TemplateJsonPath -Raw

if ($json -match "BASE64_ENCODED_USER_DATA_HERE") {
    $json = $json -replace "BASE64_ENCODED_USER_DATA_HERE", $encodedUserData
} else {
    $jsonObject = $json | ConvertFrom-Json
    $jsonObject.UserData = $encodedUserData
    $json = $jsonObject | ConvertTo-Json -Depth 20
}

Set-Content -Path $TempJsonPath -Value $json -Encoding ascii

Write-Host "Checking current latest launch template version..."
$latestVersion = aws ec2 describe-launch-templates `
  --launch-template-names $LaunchTemplateName `
  --query "LaunchTemplates[0].LatestVersionNumber" `
  --output text

if (-not $latestVersion -or $latestVersion -eq "None") {
    throw "Could not determine latest version for launch template: $LaunchTemplateName"
}

$newDescription = "v$([int]$latestVersion + 1)"
Write-Host "Creating new launch template version: $newDescription"

$createResult = aws ec2 create-launch-template-version `
  --launch-template-name $LaunchTemplateName `
  --version-description $newDescription `
  --launch-template-data file://$TempJsonPath

if ($LASTEXITCODE -ne 0) {
    throw "Failed to create launch template version."
}

Remove-Item $TempJsonPath -ErrorAction SilentlyContinue

$newLatest = aws ec2 describe-launch-templates `
  --launch-template-names $LaunchTemplateName `
  --query "LaunchTemplates[0].LatestVersionNumber" `
  --output text

Write-Host "Launch template updated successfully."
Write-Host "Latest version is now: $newLatest"