# -----------------------------------------------------------------------------
# launch_gpu.ps1
#
# Launches a GPU EC2 instance from Launch Template and prepares SSH access.
#
# Requirements:
# - Must provide KeyFile explicitly
#
# Usage:
#   .\launch_gpu.ps1 -KeyFile "C:\path\to\thesis-ec2-key.pem"
#   .\launch_gpu.ps1 -KeyFile "C:\path\to\thesis-ec2-key.pem" -AutoSSH
# -----------------------------------------------------------------------------

param(
    [Parameter(Mandatory = $true)]
    [string]$KeyFile,

    [switch]$AutoSSH
)

$LaunchTemplateName = "system2-gpu-template"
$Version = '$Latest'

if (-not (Test-Path $KeyFile)) {
    Write-Error "Key file not found: $KeyFile"
    exit 1
}

Write-Host "Using key: $KeyFile"
Write-Host "Launching EC2 instance..."

$instanceId = aws ec2 run-instances `
  --launch-template LaunchTemplateName=$LaunchTemplateName,Version=$Version `
  --query "Instances[0].InstanceId" `
  --output text

if (-not $instanceId) {
    Write-Error "Failed to launch instance"
    exit 1
}

Write-Host "Instance ID: $instanceId"
Write-Host "Waiting for instance to be running..."

aws ec2 wait instance-running --instance-ids $instanceId

Write-Host "Fetching public IP..."

$publicIp = aws ec2 describe-instances `
  --instance-ids $instanceId `
  --query "Reservations[0].Instances[0].PublicIpAddress" `
  --output text

if (-not $publicIp -or $publicIp -eq "None") {
    Write-Error "Could not retrieve public IP"
    exit 1
}

Write-Host "Public IP: $publicIp"

$sshCmd = "ssh -i `"$KeyFile`" ec2-user@$publicIp"

Write-Host ""
Write-Host "SSH command:"
Write-Host $sshCmd
Write-Host ""

if ($AutoSSH) {
    Write-Host "Connecting via SSH..."
    Start-Sleep -Seconds 5
    Invoke-Expression $sshCmd
}