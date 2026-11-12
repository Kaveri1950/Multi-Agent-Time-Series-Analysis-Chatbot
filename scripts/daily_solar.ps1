# Daily solar collection: archive NOAA/Hesperia data, then issue and log a sunspot forecast.
# Registered as the Windows task "BTP daily solar"; output goes to logs/daily_solar.log.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"
$py = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "logs\daily_solar.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

"==== $(Get-Date -Format s) ====" | Out-File -Append -Encoding utf8 $log
& $py -c "from solar.flare_data import harvest_daily; print(harvest_daily(verbose=False))" 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
& $py -m solar.sunspots --forecast 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
