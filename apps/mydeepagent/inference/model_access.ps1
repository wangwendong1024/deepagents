<#
.SYNOPSIS
Install and maintain the current user's private MyDeepAgent SSH forwards.
.EXAMPLE
.\model_access.ps1 -Mode Install
.EXAMPLE
.\model_access.ps1 -Mode Status
#>
[CmdletBinding()]
param(
    [ValidateSet('Install', 'Status', 'Repair', 'Uninstall', 'Run')]
    [string]$Mode = 'Status',
    [string]$ServerAddress = '192.168.22.59',
    [ValidateRange(1, 65535)]
    [int]$ServerPort = 2225,
    [string]$SshUser = 'root',
    [string]$KnownHostsFile = (Join-Path $env:USERPROFILE '.ssh\known_hosts'),
    [string]$IdentityFile = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Owner = 'MyDeepAgent managed model access v1'
$TaskName = 'MyDeepAgent-ModelAccess'
$AliasName = 'mydeepagent-model-access'
$CurrentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
$MutexSuffix = $CurrentUser.User.Value
$InstallRoot = Join-Path (Join-Path $env:ProgramData 'MyDeepAgent\ModelAccess') $MutexSuffix
$InstalledScript = Join-Path $InstallRoot 'model_access.ps1'
$SettingsFile = Join-Path $InstallRoot 'settings.json'
$SshConfig = Join-Path $InstallRoot 'ssh_config'
$SshLog = Join-Path $InstallRoot 'ssh.log'
$SshPath = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
$PowerShellPath = Join-Path $env:WINDIR 'System32\WindowsPowerShell\v1.0\powershell.exe'
$Ports = @(18080, 18081, 18082, 18083)
$ChildArguments = '-F "' + $SshConfig + '" -N -T ' + $AliasName
$TaskArguments = '-F "' + $SshConfig + '" -E "' + $SshLog + '" -N -T ' + $AliasName
$SourceScript = $PSCommandPath

function Assert-RegularPath([string]$Path, [switch]$Directory) {
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Refusing a reparse point at $Path"
    }
    if ($Directory.IsPresent -ne $item.PSIsContainer) {
        throw "Unexpected path type at $Path"
    }
}

function Read-Settings {
    Assert-RegularPath $InstallRoot -Directory
    Assert-RegularPath $SettingsFile
    if (-not (Test-Path -LiteralPath $SettingsFile -PathType Leaf)) { return $null }
    $settings = Get-Content -LiteralPath $SettingsFile -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($settings.Owner -ne $Owner) { throw 'The access directory belongs to another installation.' }
    return $settings
}

function Get-OwnedTask {
    $task = Get-ScheduledTask -TaskName $TaskName -TaskPath '\' -ErrorAction SilentlyContinue
    if ($null -ne $task) {
        $legacyArguments = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File "' + $InstalledScript + '" -Mode Run'
        $native = $task.Actions[0].Execute -eq $SshPath -and $task.Actions[0].Arguments -eq $TaskArguments
        $legacy = $task.Actions[0].Execute -eq $PowerShellPath -and $task.Actions[0].Arguments -eq $legacyArguments
        if ($task.Description -ne $Owner -or @($task.Actions).Count -ne 1 -or -not ($native -or $legacy)) {
            throw "The task $TaskName is not owned by this installer."
        }
    }
    return $task
}

function Get-OwnedSsh {
    return @(Get-CimInstance Win32_Process -Filter "Name='ssh.exe'" | Where-Object {
        $_.ExecutablePath -eq $SshPath -and $null -ne $_.CommandLine -and
        ($_.CommandLine.TrimEnd().EndsWith($ChildArguments, [StringComparison]::OrdinalIgnoreCase) -or
            $_.CommandLine.TrimEnd().EndsWith($TaskArguments, [StringComparison]::OrdinalIgnoreCase))
    })
}

function Get-Listeners {
    return @(Get-NetTCPConnection -State Listen -LocalPort $Ports -ErrorAction SilentlyContinue)
}

function Assert-NoForeignListeners {
    $owned = @(Get-OwnedSsh | ForEach-Object { [int]$_.ProcessId })
    $foreign = @(Get-Listeners | Where-Object { $owned -notcontains [int]$_.OwningProcess })
    if ($foreign.Count -gt 0) {
        $numbers = ($foreign | Select-Object -ExpandProperty LocalPort -Unique) -join ', '
        throw "Local model ports are already occupied by another process: $numbers"
    }
}

function Show-Status {
    $settings = Read-Settings
    $task = Get-OwnedTask
    $owned = @(Get-OwnedSsh | ForEach-Object { [int]$_.ProcessId })
    $listeners = @(Get-Listeners)
    $ready = @($Ports | Where-Object {
        $port = $_
        @($listeners | Where-Object {
            $_.LocalAddress -eq '127.0.0.1' -and $_.LocalPort -eq $port -and
            $owned -contains [int]$_.OwningProcess
        }).Count -gt 0
    })
    [PSCustomObject]@{
        Installed = $null -ne $settings -and $null -ne $task
        TaskName = $TaskName
        TaskState = $(if ($null -ne $task) { [string]$task.State } else { 'Absent' })
        ReadyPorts = $ready
        ExpectedPorts = $Ports
        SshProcessIds = $owned
        Directory = $InstallRoot
        Log = $SshLog
    }
}

function Stop-OwnedProcesses {
    $task = Get-OwnedTask
    if ($null -ne $task -and [string]$task.State -in @('Running', 'Queued')) {
        Stop-ScheduledTask -TaskName $TaskName -TaskPath '\'
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            $state = (Get-OwnedTask).State
            if ($state -ne 'Running') { break }
            Start-Sleep -Milliseconds 250
        }
    }
    foreach ($child in @(Get-OwnedSsh)) {
        Stop-Process -Id $child.ProcessId -ErrorAction SilentlyContinue
    }
}

function Resolve-SshFile([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).ProviderPath
    Assert-RegularPath $resolved
    if ($resolved.IndexOfAny([char[]]'"%') -ge 0 -or $resolved -match '[\r\n]') {
        throw 'SSH file paths must not contain quotes, percent signs, or newlines.'
    }
    return $resolved.Replace('\', '/')
}

function Set-PrivateDirectoryAcl {
    $acl = New-Object Security.AccessControl.DirectorySecurity
    $acl.SetAccessRuleProtection($true, $false)
    $acl.SetOwner($CurrentUser.User)
    $inheritance = [Security.AccessControl.InheritanceFlags]'ContainerInherit, ObjectInherit'
    foreach ($sid in @($CurrentUser.User, (New-Object Security.Principal.SecurityIdentifier('S-1-5-18')))) {
        $rule = New-Object Security.AccessControl.FileSystemAccessRule(
            $sid, [Security.AccessControl.FileSystemRights]::FullControl, $inheritance,
            [Security.AccessControl.PropagationFlags]::None, [Security.AccessControl.AccessControlType]::Allow
        )
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $InstallRoot -AclObject $acl
}

function Build-SshConfiguration($Settings) {
    if ($Settings.ServerAddress -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]{0,253}$' -or
        $Settings.SshUser -notmatch '^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$' -or
        [int]$Settings.ServerPort -lt 1 -or [int]$Settings.ServerPort -gt 65535) {
        throw 'Invalid SSH server configuration.'
    }
    $knownHosts = Resolve-SshFile $Settings.KnownHostsFile
    $lookup = $Settings.ServerAddress
    if ([int]$Settings.ServerPort -ne 22) { $lookup = '[' + $lookup + ']:' + $Settings.ServerPort }
    $keygen = Join-Path (Split-Path -Parent $SshPath) 'ssh-keygen.exe'
    $null = & $keygen -F $lookup -f $knownHosts
    if ($LASTEXITCODE -ne 0) { throw "The trusted known_hosts entry is missing for $lookup" }
    $lines = @(
        ('# ' + $Owner)
        ('Host ' + $AliasName)
        ('    HostName ' + $Settings.ServerAddress)
        ('    Port ' + $Settings.ServerPort)
        ('    User ' + $Settings.SshUser)
        ('    UserKnownHostsFile "' + $knownHosts + '"')
        '    BatchMode yes'
        '    PreferredAuthentications publickey'
        '    PasswordAuthentication no'
        '    KbdInteractiveAuthentication no'
        '    StrictHostKeyChecking yes'
        '    UpdateHostKeys no'
        '    ProxyCommand none'
        '    ProxyJump none'
        '    ForwardAgent no'
        '    ForwardX11 no'
        '    ExitOnForwardFailure yes'
        '    ConnectTimeout 8'
        '    ConnectionAttempts 1'
        '    ServerAliveInterval 15'
        '    ServerAliveCountMax 3'
        '    LogLevel ERROR'
    )
    if ($Settings.IdentityFile) {
        $identity = Resolve-SshFile $Settings.IdentityFile
        $lines += '    IdentityFile "' + $identity + '"'
        $lines += '    IdentitiesOnly yes'
        $lines += '    IdentityAgent none'
    }
    foreach ($port in $Ports) { $lines += "    LocalForward 127.0.0.1:$port 127.0.0.1:$port" }
    return ($lines -join "`n") + "`n"
}

function Install-Access {
    $previous = Read-Settings
    $settings = [ordered]@{
        Owner = $Owner
        ServerAddress = $ServerAddress
        ServerPort = $ServerPort
        SshUser = $SshUser
        KnownHostsFile = $KnownHostsFile
        IdentityFile = $IdentityFile
    }
    if ($null -ne $previous) {
        foreach ($name in @('ServerAddress', 'ServerPort', 'SshUser', 'KnownHostsFile', 'IdentityFile')) {
            if (-not $script:PSBoundParameters.ContainsKey($name)) { $settings[$name] = $previous.$name }
        }
    } elseif (Test-Path -LiteralPath $InstallRoot) {
        $provisionedKey = Join-Path $InstallRoot 'model-access_ed25519'
        $provisionedHosts = Join-Path $InstallRoot 'known_hosts'
        $entries = @(Get-ChildItem -LiteralPath $InstallRoot -Force)
        $unexpected = @($entries | Where-Object {
            $_.Name -notin @('model-access_ed25519', 'model-access_ed25519.pub', 'known_hosts')
        })
        if ($entries.Count -gt 0 -and
            ($unexpected.Count -gt 0 -or $IdentityFile -ne $provisionedKey -or
                ((Test-Path -LiteralPath $provisionedHosts) -and $KnownHostsFile -ne $provisionedHosts))) {
            throw 'The unrecognized access directory is not empty.'
        }
    }
    $configuration = Build-SshConfiguration ([PSCustomObject]$settings)
    if (-not $settings.IdentityFile) { throw 'A dedicated IdentityFile is required for the noninteractive task.' }
    $null = Get-OwnedTask
    Assert-NoForeignListeners
    foreach ($path in @($InstalledScript, $SshConfig, $SshLog)) { Assert-RegularPath $path }
    Stop-OwnedProcesses
    $null = New-Item -ItemType Directory -Path $InstallRoot -Force
    Set-PrivateDirectoryAcl
    $encoding = New-Object Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($SettingsFile, (($settings | ConvertTo-Json) -replace "`r`n", "`n") + "`n", $encoding)
    [IO.File]::WriteAllText($SshConfig, $configuration, $encoding)
    if ($SourceScript -ne $InstalledScript) { Copy-Item -LiteralPath $SourceScript -Destination $InstalledScript -Force }
    $null = & $SshPath -F $SshConfig -G $AliasName
    if ($LASTEXITCODE -ne 0) { throw 'The generated SSH configuration was rejected.' }
    $action = New-ScheduledTaskAction -Execute $SshPath -Argument $TaskArguments -WorkingDirectory (Join-Path $env:WINDIR 'System32')
    $triggers = @(
        (New-ScheduledTaskTrigger -AtStartup)
        (New-ScheduledTaskTrigger -AtLogOn -User $CurrentUser.Name)
        (New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) -RepetitionInterval (New-TimeSpan -Minutes 1))
    )
    $principal = New-ScheduledTaskPrincipal -UserId $CurrentUser.Name -LogonType S4U -RunLevel Limited
    $options = New-ScheduledTaskSettingsSet -Hidden -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
    $null = Register-ScheduledTask -TaskName $TaskName -TaskPath '\' -Action $action -Trigger $triggers `
        -Principal $principal -Settings $options -Description $Owner -Force
    Start-ScheduledTask -TaskName $TaskName -TaskPath '\'
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $status = Show-Status
        if (@($status.ReadyPorts).Count -eq 4) { return $status }
        Start-Sleep -Seconds 1
    }
    Show-Status
    throw "The tunnel has not bound all four ports. See $SshLog"
}

function Run-Access {
    if ($null -eq (Read-Settings)) { throw 'Install the access configuration first.' }
    Assert-RegularPath $SshConfig
    $guard = New-Object Threading.Mutex($false, ('Local\MyDeepAgent-ModelAccess-Run-' + $MutexSuffix))
    if (-not $guard.WaitOne(0)) { $guard.Dispose(); return }
    $child = $null
    try {
        foreach ($orphan in @(Get-OwnedSsh)) {
            Stop-Process -Id $orphan.ProcessId -ErrorAction SilentlyContinue
        }
        $delay = 5
        while ($true) {
            Assert-NoForeignListeners
            $started = [DateTime]::UtcNow
            $child = Start-Process -FilePath $SshPath -ArgumentList $ChildArguments -WindowStyle Hidden `
                -RedirectStandardError $SshLog -PassThru
            $child.WaitForExit()
            $child.Dispose()
            $child = $null
            if (([DateTime]::UtcNow - $started).TotalSeconds -ge 60) { $delay = 5 }
            Start-Sleep -Seconds $delay
            $delay = [Math]::Min(30, $delay * 2)
        }
    } finally {
        if ($null -ne $child) {
            if (-not $child.HasExited) { $child.Kill() }
            $child.Dispose()
        }
        $guard.ReleaseMutex()
        $guard.Dispose()
    }
}

if (-not (Test-Path -LiteralPath $SshPath -PathType Leaf)) { throw 'Windows OpenSSH Client is not installed.' }
if ($Mode -eq 'Run') { Run-Access; return }
if ($Mode -eq 'Status') { Show-Status; return }

$maintenance = New-Object Threading.Mutex($false, ('Local\MyDeepAgent-ModelAccess-Maintenance-' + $MutexSuffix))
if (-not $maintenance.WaitOne(10000)) { $maintenance.Dispose(); throw 'Another access maintenance operation is running.' }
try {
    if ($Mode -eq 'Uninstall') {
        $null = Read-Settings
        $task = Get-OwnedTask
        Stop-OwnedProcesses
        if ($null -ne $task) { Unregister-ScheduledTask -TaskName $TaskName -TaskPath '\' -Confirm:$false }
        if (Test-Path -LiteralPath $SettingsFile) {
            foreach ($path in @($SshConfig, $SshLog, $InstalledScript, $SettingsFile)) {
                Assert-RegularPath $path
                if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force }
            }
            if (@(Get-ChildItem -LiteralPath $InstallRoot -Force).Count -eq 0) { Remove-Item -LiteralPath $InstallRoot }
        }
        'MyDeepAgent model access uninstalled.'
    } else {
        Install-Access
    }
} finally {
    $maintenance.ReleaseMutex()
    $maintenance.Dispose()
}
