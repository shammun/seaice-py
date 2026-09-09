# Mirror the parts of seaice-py that Colab needs into Google Drive (Drive for Desktop must be installed).
# Usage:  powershell -ExecutionPolicy Bypass -File tools/sync_to_drive.ps1 -DriveDir "G:\My Drive\seaice-py"
param([string]$DriveDir = "G:\My Drive\seaice-py")
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
foreach ($d in @("notebooks", "seaice", "knowledge", "data\book", "data\online", "data\synthetic")) {
    $src = Join-Path $root $d; $dst = Join-Path $DriveDir $d
    if (Test-Path $src) { robocopy $src $dst /MIR /XD __pycache__ .ipynb_checkpoints /XF executed_*.ipynb *.pyc /NFL /NDL /NJH /NJS | Out-Null; Write-Host "synced $d" }
}
Copy-Item (Join-Path $root "requirements-colab.txt") $DriveDir -Force
Copy-Item (Join-Path $root "book.yaml") $DriveDir -Force
Write-Host "Done -> $DriveDir  (data\manual is never synced from here; put manual downloads directly in Drive)"
