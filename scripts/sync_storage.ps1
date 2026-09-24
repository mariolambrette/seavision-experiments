<#
.SYNOPSIS
    Sync the SeaVision storage trees between the NAS and the workstation.

.DESCRIPTION
    Supersedes mirror_to_nas.ps1, which only knew about shards and embeddings.

    Each tree has ONE authoritative copy and ONE sync direction. The direction
    is a property of the tree, not of the invocation, so there is no way to run
    this script and push the wrong way:

      raw/         NAS is authoritative.  NAS -> D:   (pull)
                   Irreplaceable. Nothing writes to it on either side. /MIR is
                   safe here because it can only ever delete from the COPY.

      collated/    D: is authoritative.   D: -> NAS   (push)
                   Derived: rebuildable from raw/ plus code plus config. /MIR
                   is used deliberately -- see the note below.

      shards/      D: is authoritative.   D: -> NAS   (push)
      embeddings/  D: is authoritative.   D: -> NAS   (push)

    WHY /MIR ON collated/ AND NOT /E
    An /E push never deletes, so the NAS would accumulate crops from
    superseded builds with nothing to distinguish them from current ones --
    which is exactly how images_orphaned/ came about. A backup you cannot
    trust is worse than one you can rebuild. /MIR keeps the two identical, and
    the cost of a wrongly-propagated deletion is a rebuild, not data.

    THE SHRINK GUARD
    The realistic /MIR disaster is not a bad flag, it is a source that is
    empty or partial -- a failed build, an unmounted drive, a half-finished
    run -- being mirrored over a good destination. Every push therefore checks
    that its source still looks populated before robocopy is allowed near the
    destination. -Strict turns that into a full count comparison, which is
    slower but catches partial loss as well as total loss.

.EXAMPLE
    .\sync_storage.ps1                        # dry run, everything
    .\sync_storage.ps1 -Task push-collated -Execute
    .\sync_storage.ps1 -Execute -Strict
#>

[CmdletBinding()]
param(
    [ValidateSet('pull-raw', 'push-collated', 'push-derived', 'all')]
    [string]$Task = 'all',

    # Dry run is the default. Nothing is written without -Execute.
    [switch]$Execute,

    # Compare full file counts before a push instead of the cheap check.
    [switch]$Strict,

    # Bypass the shrink guard. Only when you know why the source shrank.
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$NAS = 'N:\marineai'
$WS  = 'D:\marineai'
$LOG = Join-Path $WS 'logs'

# Minimum file counts a source must still have for a push to be allowed.
# Set well below the real figures: this is a "did the source vanish" test,
# not a "is the source exactly right" test.
$MinFiles = @{
    'collated'   = 1000000     # 1,471,937 at the time of writing
    'shards'     = 1
    'embeddings' = 1
}

function Test-MinFiles {
    param([string]$Path, [int]$Min)
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    if ($Min -le 0) { return $true }
    $n = 0
    foreach ($f in [System.IO.Directory]::EnumerateFiles($Path, '*', 'AllDirectories')) {
        $n++
        if ($n -ge $Min) { return $true }
    }
    return $false
}

function Measure-Tree {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return [pscustomobject]@{ Files = 0; Bytes = 0 } }
    $files = 0; $bytes = 0
    foreach ($f in [System.IO.Directory]::EnumerateFiles($Path, '*', 'AllDirectories')) {
        $files++; $bytes += (New-Object System.IO.FileInfo $f).Length
    }
    [pscustomobject]@{ Files = $files; Bytes = $bytes }
}

function Invoke-Sync {
    param(
        [string]$Name,
        [string]$Source,
        [string]$Dest,
        [switch]$Mirror,
        [int]$MinSourceFiles = 0,
        [switch]$GuardSource        # only pushes need guarding
    )

    Write-Host ""
    Write-Host ("=" * 72)
    Write-Host "$Name"
    Write-Host ("  {0}" -f $Source)
    Write-Host ("  -> {0}" -f $Dest)
    Write-Host ("=" * 72)

    if (-not (Test-Path -LiteralPath $Source)) {
        Write-Host "  SKIP: source does not exist" -ForegroundColor Yellow
        return
    }

    if ($GuardSource -and -not $Force) {
        if (-not (Test-MinFiles -Path $Source -Min $MinSourceFiles)) {
            Write-Host ("  REFUSED: source holds fewer than {0:N0} files. " -f $MinSourceFiles) -ForegroundColor Red
            Write-Host "  A mirror from a partial source would delete the good copy." -ForegroundColor Red
            Write-Host "  Fix the source, or pass -Force if you know why it shrank." -ForegroundColor Red
            return
        }
        if ($Strict -and (Test-Path -LiteralPath $Dest)) {
            Write-Host "  counting both sides (-Strict) ..."
            $s = Measure-Tree $Source
            $d = Measure-Tree $Dest
            Write-Host ("  source {0,12:N0} files  {1,10:N2} GB" -f $s.Files, ($s.Bytes / 1GB))
            Write-Host ("  dest   {0,12:N0} files  {1,10:N2} GB" -f $d.Files, ($d.Bytes / 1GB))
            if ($d.Files -gt 0 -and $s.Files -lt ($d.Files * 0.9)) {
                Write-Host "  REFUSED: source has lost more than 10% of the destination's files." -ForegroundColor Red
                Write-Host "  Pass -Force only if that loss was intended." -ForegroundColor Red
                return
            }
        }
    }

    $mode = if ($Mirror) { '/MIR' } else { '/E' }
    # NOT $args -- that is a PowerShell automatic variable and assigning to it
    # inside a function is at best confusing and at worst an error.
    $rcArgs = @($Source, $Dest, $mode, '/MT:32', '/R:2', '/W:5', '/NFL', '/NDL', '/NP')
    if (-not $Execute) { $rcArgs += '/L' }
    New-Item -ItemType Directory -Force -Path $LOG | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd'
    $rcArgs += "/LOG+:$LOG\sync_$($Name -replace '[^a-zA-Z0-9]','_')_$stamp.log"

    if (-not $Execute) { Write-Host "  DRY RUN (add -Execute to apply)" -ForegroundColor Cyan }
    robocopy @rcArgs | Out-Null
    $code = $LASTEXITCODE

    # robocopy: 0-7 are success states, 8+ are failures.
    if ($code -ge 8) {
        Write-Host ("  ROBOCOPY FAILED (exit {0}) -- see the log" -f $code) -ForegroundColor Red
    } else {
        Write-Host ("  ok (exit {0})" -f $code) -ForegroundColor Green
    }
}

if ($Task -in @('pull-raw', 'all')) {
    # NAS -> D:. /MIR is safe in this direction: it can only delete from the
    # copy. The reverse is deliberately not implemented anywhere in this file.
    Invoke-Sync -Name 'raw (NAS to workstation)' `
                -Source "$NAS\dataset\raw" -Dest "$WS\dataset\raw" -Mirror
}

if ($Task -in @('push-collated', 'all')) {
    Invoke-Sync -Name 'collated (workstation to NAS)' `
                -Source "$WS\dataset\collated" -Dest "$NAS\dataset\collated" `
                -Mirror -GuardSource -MinSourceFiles $MinFiles['collated']
}

if ($Task -in @('push-derived', 'all')) {
    Invoke-Sync -Name 'shards (workstation to NAS)' `
                -Source "$WS\classification-experiments\shards" `
                -Dest "$NAS\classification-experiments\shards" `
                -Mirror -GuardSource -MinSourceFiles $MinFiles['shards']
    Invoke-Sync -Name 'embeddings (workstation to NAS)' `
                -Source "$WS\classification-experiments\embeddings" `
                -Dest "$NAS\classification-experiments\embeddings" `
                -Mirror -GuardSource -MinSourceFiles $MinFiles['embeddings']
}

Write-Host ""
if (-not $Execute) {
    Write-Host "Dry run only. Re-run with -Execute to apply." -ForegroundColor Cyan
}
Write-Host "Logs: $LOG"