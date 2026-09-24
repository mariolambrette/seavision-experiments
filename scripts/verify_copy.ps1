<#
.SYNOPSIS
    Verify the workstation copy against the NAS original.

.DESCRIPTION
    robocopy reporting exit 0 means it believes it copied everything. It does
    not mean the bytes on the far side are the same bytes. This checks three
    things, cheapest first:

      1. file count per tree
      2. total bytes per tree
      3. SHA-256 of a random sample of files, on both sides

    (3) is the only one that can catch silent corruption, and a sample is the
    only affordable version of it across 281 GB. The sample is seeded, so a
    re-run checks the same files and a disagreement is reproducible.

    Run this BEFORE pointing any config at the copy. A config change is cheap
    to make and expensive to discover was premature.

.EXAMPLE
    .\verify_copy.ps1
    .\verify_copy.ps1 -Sample 500 -Seed 42
#>

[CmdletBinding()]
param(
    [string]$Nas = 'N:\marineai\dataset',
    [string]$Workstation = 'D:\marineai\dataset',
    [string[]]$Trees = @('raw', 'collated'),
    [int]$Sample = 200,
    [int]$Seed = 0
)

$ErrorActionPreference = 'Stop'
$failures = @()

function Measure-Tree {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    $files = 0; $bytes = 0
    foreach ($f in [System.IO.Directory]::EnumerateFiles($Path, '*', 'AllDirectories')) {
        $files++; $bytes += (New-Object System.IO.FileInfo $f).Length
    }
    [pscustomobject]@{ Files = $files; Bytes = $bytes }
}

Write-Host ""
Write-Host "COUNTS AND BYTES"
Write-Host ("-" * 78)
Write-Host ("{0,-12}{1,14}{2,14}{3,12}{4,12}" -f 'tree', 'NAS files', 'D: files', 'NAS GB', 'D: GB')

foreach ($t in $Trees) {
    $a = Measure-Tree (Join-Path $Nas $t)
    $b = Measure-Tree (Join-Path $Workstation $t)
    if ($null -eq $a) { Write-Host "  $t : missing on the NAS"; $failures += "$t missing on NAS"; continue }
    if ($null -eq $b) { Write-Host "  $t : missing on D:";     $failures += "$t missing on D:";  continue }
    Write-Host ("{0,-12}{1,14:N0}{2,14:N0}{3,12:N2}{4,12:N2}" -f `
        $t, $a.Files, $b.Files, ($a.Bytes / 1GB), ($b.Bytes / 1GB))

    # collated/ is EXPECTED to differ: pre_migrate_backup was moved out of it
    # on the NAS and build_manifest.json was deleted, so an exact match there
    # would actually mean the housekeeping had not happened.
    if ($a.Files -ne $b.Files) {
        $note = if ($t -eq 'collated') { ' (expected: housekeeping removed items from the NAS side)' } else { '' }
        Write-Host ("  ! {0}: file counts differ by {1:N0}{2}" -f $t, [math]::Abs($a.Files - $b.Files), $note) -ForegroundColor Yellow
        if ($t -ne 'collated') { $failures += "$t file count mismatch" }
    }
    if ($a.Bytes -ne $b.Bytes -and $t -ne 'collated') {
        Write-Host ("  ! {0}: byte totals differ by {1:N0}" -f $t, [math]::Abs($a.Bytes - $b.Bytes)) -ForegroundColor Yellow
        $failures += "$t byte mismatch"
    }
}

Write-Host ""
Write-Host "SAMPLED HASH CHECK ($Sample files per tree, seed $Seed)"
Write-Host ("-" * 78)

$rng = [System.Random]::new($Seed)
foreach ($t in $Trees) {
    $root = Join-Path $Nas $t
    if (-not (Test-Path -LiteralPath $root)) { continue }

    # Reservoir sample, so the whole tree is never held in memory.
    $res = New-Object System.Collections.ArrayList
    $i = 0
    foreach ($f in [System.IO.Directory]::EnumerateFiles($root, '*', 'AllDirectories')) {
        if ($res.Count -lt $Sample) { [void]$res.Add($f) }
        else {
            $j = $rng.Next(0, $i + 1)
            if ($j -lt $Sample) { $res[$j] = $f }
        }
        $i++
    }

    $checked = 0; $bad = 0; $missing = 0
    foreach ($src in $res) {
        $rel = $src.Substring($root.Length).TrimStart('\')
        $dst = Join-Path (Join-Path $Workstation $t) $rel
        if (-not (Test-Path -LiteralPath $dst)) {
            $missing++
            Write-Host "  MISSING on D:  $rel" -ForegroundColor Red
            continue
        }
        $h1 = (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash
        $h2 = (Get-FileHash -LiteralPath $dst -Algorithm SHA256).Hash
        $checked++
        if ($h1 -ne $h2) {
            $bad++
            Write-Host "  HASH MISMATCH  $rel" -ForegroundColor Red
        }
    }
    Write-Host ("  {0,-12} {1,5:N0} of {2:N0} sampled matched, {3} mismatched, {4} missing" -f `
        $t, ($checked - $bad), $checked, $bad, $missing)
    if ($bad -gt 0)     { $failures += "$t hash mismatches" }
    if ($missing -gt 0) { $failures += "$t missing files" }
}

Write-Host ""
if ($failures.Count -gt 0) {
    Write-Host ("VERIFY FAILED: " + ($failures -join '; ')) -ForegroundColor Red
    Write-Host "Do NOT repoint any config until this is resolved." -ForegroundColor Red
    exit 1
}
Write-Host "Copy verified. Safe to repoint the configs." -ForegroundColor Green