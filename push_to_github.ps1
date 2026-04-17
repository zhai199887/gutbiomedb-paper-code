# Push GutBiomeDB paper-code to a fresh, standalone GitHub repository.
#
# Prereqs (one-time):
#   1. Create an empty repo on github.com:
#        Name:        gutbiomedb-paper-code
#        Owner:       zhai199887
#        Visibility:  Public
#        DO NOT initialise with README / LICENSE / .gitignore (we already have them)
#   2. Make sure `git` is on PATH (test: `git --version`)
#
# Run from this directory:
#   powershell -ExecutionPolicy Bypass -File .\push_to_github.ps1

$ErrorActionPreference = 'Stop'

$RepoDir  = 'E:\tasks\gutbiomedb-paper-code'
$RepoUrl  = 'https://github.com/zhai199887/gutbiomedb-paper-code.git'
$Branch   = 'main'

Set-Location $RepoDir
Write-Host "[*] Working dir: $RepoDir" -ForegroundColor Cyan

if (-not (Test-Path '.git')) {
    Write-Host "[*] git init" -ForegroundColor Cyan
    git init
    git branch -M $Branch
}

Write-Host "[*] git add ." -ForegroundColor Cyan
git add .

$pending = git status --porcelain
if ([string]::IsNullOrWhiteSpace($pending)) {
    Write-Host "[!] Nothing to commit." -ForegroundColor Yellow
} else {
    Write-Host "[*] git commit" -ForegroundColor Cyan
    git commit -m "Initial commit: GutBiomeDB manuscript figure-generation code"
}

$existingRemote = git remote
if ($existingRemote -notcontains 'origin') {
    Write-Host "[*] git remote add origin $RepoUrl" -ForegroundColor Cyan
    git remote add origin $RepoUrl
} else {
    Write-Host "[*] remote 'origin' already exists, updating URL" -ForegroundColor Cyan
    git remote set-url origin $RepoUrl
}

Write-Host "[*] git push -u origin $Branch" -ForegroundColor Cyan
git push -u origin $Branch

Write-Host ""
Write-Host "[OK] Pushed. Open: https://github.com/zhai199887/gutbiomedb-paper-code" -ForegroundColor Green
