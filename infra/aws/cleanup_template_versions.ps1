# -----------------------------------------------------------------------------
# cleanup_template_versions.ps1
#
# Deletes old EC2 Launch Template versions while keeping:
# - the default version
# - the latest version
# - the most recent N versions
#
# Usage:
#   .\cleanup_template_versions.ps1
# -----------------------------------------------------------------------------

$LaunchTemplateName = "system2-gpu-template"
$KeepRecent = 5

Write-Host "Reading launch template versions..."
$versionsJson = aws ec2 describe-launch-template-versions `
  --launch-template-name $LaunchTemplateName `
  --versions All `
  --query "{Default:LaunchTemplateVersions[?DefaultVersion==\`true\`].VersionNumber | [0], Latest:LaunchTemplateVersions[?LatestVersion==\`true\`].VersionNumber | [0], Versions:LaunchTemplateVersions[].VersionNumber}" `
  --output json

$data = $versionsJson | ConvertFrom-Json

$defaultVersion = [int]$data.Default
$latestVersion = [int]$data.Latest
$allVersions = @($data.Versions | ForEach-Object { [int]$_ } | Sort-Object)

$recentVersions = $allVersions | Sort-Object -Descending | Select-Object -First $KeepRecent
$keep = @($defaultVersion, $latestVersion) + $recentVersions
$keep = $keep | Sort-Object -Unique

$toDelete = $allVersions | Where-Object { $_ -notin $keep }

Write-Host "Default version: $defaultVersion"
Write-Host "Latest version: $latestVersion"
Write-Host "Keeping versions: $($keep -join ', ')"

if ($toDelete.Count -eq 0) {
    Write-Host "No old versions to delete."
    exit 0
}

Write-Host "Deleting old versions: $($toDelete -join ', ')"

aws ec2 delete-launch-template-versions `
  --launch-template-name $LaunchTemplateName `
  --versions ($toDelete | ForEach-Object { "$_" })

Write-Host "Cleanup complete."