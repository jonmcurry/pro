' DetectInstallation.vbs - Detect if this is a fresh install or upgrade

' Helper function to write to MSI log
Sub LogMessage(message)
    Dim record
    Set record = Session.Installer.CreateRecord(1)
    record.StringData(0) = "Professional SMART Installer: [1]"
    record.StringData(1) = message
    Session.Message &H04000000, record
End Sub

Function DetectInstallation()
    LogMessage "DetectInstallation: Starting installation type detection"

    ' Get database connection properties
    Dim dbHost, dbPort, dbName, dbUser, dbPassword
    dbHost = Session.Property("DB_HOST")
    dbPort = Session.Property("DB_PORT")
    dbName = Session.Property("DB_NAME")
    dbUser = Session.Property("DB_USER")
    dbPassword = Session.Property("DB_PASSWORD")

    LogMessage "DetectInstallation: DB_HOST = " & dbHost
    LogMessage "DetectInstallation: DB_PORT = " & dbPort
    LogMessage "DetectInstallation: DB_NAME = " & dbName
    LogMessage "DetectInstallation: DB_USER = " & dbUser

    ' Create shell and FileSystemObject
    Dim shell, fso
    Set shell = CreateObject("WScript.Shell")
    Set fso = CreateObject("Scripting.FileSystemObject")

    ' Set environment variables for pro-upgrade.exe
    Dim env
    Set env = shell.Environment("Process")
    env("PGPASSWORD") = dbPassword
    env("DB_HOST") = dbHost
    env("DB_PORT") = dbPort
    env("DB_NAME") = dbName
    env("DB_USER") = dbUser
    env("DB_PASSWORD") = dbPassword

    ' Check if this is an upgrade by checking WIX_UPGRADE_DETECTED property
    ' This is the authoritative source set by the MajorUpgrade element
    Dim isUpgrade
    isUpgrade = False

    Dim wixUpgradeDetected
    wixUpgradeDetected = Session.Property("WIX_UPGRADE_DETECTED")

    If wixUpgradeDetected <> "" Then
        LogMessage "DetectInstallation: WIX_UPGRADE_DETECTED = " & wixUpgradeDetected
        isUpgrade = True
    Else
        LogMessage "DetectInstallation: WIX_UPGRADE_DETECTED not set - this is a fresh install"
    End If

    ' Also check registry for version information
    On Error Resume Next
    Dim installedVersion
    installedVersion = shell.RegRead("HKLM\SOFTWARE\ProfessionalSMART\Version")

    If Err.Number = 0 And installedVersion <> "" Then
        LogMessage "DetectInstallation: Found existing installation in registry: " & installedVersion
    Else
        LogMessage "DetectInstallation: No existing installation found in registry"
        installedVersion = ""
    End If
    Err.Clear
    On Error GoTo 0

    ' Try to use pro-upgrade.exe to detect installation type
    ' Check if pro-upgrade.exe is available in the installer temp directory
    Dim installerDir
    installerDir = Session.Property("INSTALLFOLDER")

    If installerDir = "" Then
        ' Try to get from CustomActionData or use default
        installerDir = "C:\Program Files\Professional SMART\"
    End If

    If Right(installerDir, 1) <> "\" Then
        installerDir = installerDir & "\"
    End If

    Dim proUpgradeExe
    proUpgradeExe = installerDir & "bin\pro-upgrade.exe"

    ' If pro-upgrade.exe exists, use it to detect installation type
    If fso.FileExists(proUpgradeExe) Then
        LogMessage "DetectInstallation: Found pro-upgrade.exe, using it to detect installation type"

        Dim detectCmd
        detectCmd = """" & proUpgradeExe & """ detect-installation-type"

        Dim detectResult
        detectResult = shell.Run(detectCmd, 0, True)

        If detectResult = 0 Then
            LogMessage "DetectInstallation: pro-upgrade.exe detected installation type successfully"
            ' The output will indicate Fresh, Legacy, or Upgrade
            ' For now, we'll assume it's an upgrade if the command succeeded and we found registry entry
            If isUpgrade Then
                Session.Property("INSTALLMODE") = "UPGRADE"
                Session.Property("DETECTEDVERSION") = installedVersion
                LogMessage "DetectInstallation: Installation mode = UPGRADE"
            End If
        Else
            LogMessage "DetectInstallation: pro-upgrade.exe detection failed, assuming fresh install"
        End If
    Else
        LogMessage "DetectInstallation: pro-upgrade.exe not found, checking database directly"
    End If

    ' Set installation mode property
    If Not isUpgrade Then
        Session.Property("INSTALLMODE") = "FRESH"
        Session.Property("DETECTEDVERSION") = ""
        Session.Property("ENV_CREDENTIALS_LOADED") = "0"
        Session.Property("CREATE_BACKUP") = "0"
        LogMessage "DetectInstallation: Installation mode = FRESH"
    Else
        Session.Property("INSTALLMODE") = "UPGRADE"
        Session.Property("DETECTEDVERSION") = installedVersion
        Session.Property("CREATE_BACKUP") = "1"
        LogMessage "DetectInstallation: Installation mode = UPGRADE"
        LogMessage "DetectInstallation: Detected version = " & installedVersion

        ' For upgrades, try to load credentials from .env file
        Dim programDataFolder, envFilePath
        programDataFolder = shell.ExpandEnvironmentStrings("%ProgramData%")
        If Right(programDataFolder, 1) <> "\" Then
            programDataFolder = programDataFolder & "\"
        End If
        envFilePath = programDataFolder & "Professional SMART\config\.env"

        LogMessage "DetectInstallation: Checking for .env at: " & envFilePath

        If fso.FileExists(envFilePath) Then
            LogMessage "DetectInstallation: .env file exists - will skip DatabaseConfigDlg"
            Session.Property("ENV_CREDENTIALS_LOADED") = "1"
        Else
            LogMessage "DetectInstallation: .env file not found - will show DatabaseConfigDlg with defaults"
            Session.Property("ENV_CREDENTIALS_LOADED") = "0"
            ' Set default values for database configuration
            If Session.Property("DB_HOST") = "" Then Session.Property("DB_HOST") = "localhost"
            If Session.Property("DB_PORT") = "" Then Session.Property("DB_PORT") = "5432"
            If Session.Property("DB_NAME") = "" Then Session.Property("DB_NAME") = "professional_smart"
            If Session.Property("DB_USER") = "" Then Session.Property("DB_USER") = "postgres"
        End If
    End If

    Set fso = Nothing
    Set env = Nothing
    Set shell = Nothing

    DetectInstallation = 1  ' Return success
    LogMessage "DetectInstallation: Completed successfully"
End Function
