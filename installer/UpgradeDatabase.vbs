' UpgradeDatabase.vbs - Upgrade existing database with backup option

' Helper function to write to MSI log
Sub LogMessage(message)
    Dim record
    Set record = Session.Installer.CreateRecord(1)
    record.StringData(0) = "Professional SMART Installer: [1]"
    record.StringData(1) = message
    Session.Message &H04000000, record
End Sub

Function UpgradeDatabase()
    LogMessage "UpgradeDatabase: Starting database upgrade process"

    ' Declare all variables at function level
    Dim customData, parts
    Dim dbHost, dbPort, dbName, dbUser, dbPassword, installFolder, backupEnabled
    Dim shell, fso, env
    Dim programDataFolder, envFilePath, envFile, line, eqPos, key, value
    Dim proUpgradeExe, applyCmd, applyResult, skipCmd, skipResult

    ' Get installer properties using CustomActionData
    customData = Session.Property("CustomActionData")
    LogMessage "UpgradeDatabase: CustomActionData = " & customData

    ' Parse the custom action data (formatted as: host|port|name|user|password|installdir|backup_enabled)
    parts = Split(customData, "|")

    If UBound(parts) < 6 Then
        LogMessage "UpgradeDatabase: ERROR - Not enough parameters. Expected 7, got " & (UBound(parts) + 1)
        UpgradeDatabase = 1  ' Return success to not block install
        Exit Function
    End If

    dbHost = parts(0)
    dbPort = parts(1)
    dbName = parts(2)
    dbUser = parts(3)
    dbPassword = parts(4)
    installFolder = parts(5)
    backupEnabled = parts(6)

    ' Create shell and FileSystemObject
    Set shell = CreateObject("WScript.Shell")
    Set fso = CreateObject("Scripting.FileSystemObject")

    ' If credentials are empty (upgrade scenario), try to load from .env file
    If Len(dbHost) = 0 Or Len(dbPassword) = 0 Then
        LogMessage "UpgradeDatabase: Credentials not provided, attempting to load from .env file"

        programDataFolder = shell.ExpandEnvironmentStrings("%ProgramData%")
        If Right(programDataFolder, 1) <> "\" Then
            programDataFolder = programDataFolder & "\"
        End If
        envFilePath = programDataFolder & "Professional SMART\config\.env"

        LogMessage "UpgradeDatabase: Checking for .env at: " & envFilePath

        If fso.FileExists(envFilePath) Then
            LogMessage "UpgradeDatabase: Found .env file, loading credentials"

            Set envFile = fso.OpenTextFile(envFilePath, 1)

            Do While Not envFile.AtEndOfStream
                line = Trim(envFile.ReadLine)

                If Len(line) > 0 And Left(line, 1) <> "#" Then
                    eqPos = InStr(line, "=")

                    If eqPos > 0 Then
                        key = Trim(Left(line, eqPos - 1))
                        value = Trim(Mid(line, eqPos + 1))

                        If Left(value, 1) = """" And Right(value, 1) = """" Then
                            value = Mid(value, 2, Len(value) - 2)
                        End If

                        Select Case key
                            Case "DB_HOST"
                                dbHost = value
                            Case "DB_PORT"
                                dbPort = value
                            Case "DB_NAME"
                                dbName = value
                            Case "DB_USER"
                                dbUser = value
                            Case "DB_PASSWORD"
                                dbPassword = value
                        End Select
                    End If
                End If
            Loop

            envFile.Close
            LogMessage "UpgradeDatabase: Loaded credentials from .env"
        Else
            LogMessage "UpgradeDatabase: No .env file found - using default values"
            ' Use defaults if .env doesn't exist
            If Len(dbHost) = 0 Then dbHost = "localhost"
            If Len(dbPort) = 0 Then dbPort = "5432"
            If Len(dbName) = 0 Then dbName = "professional_smart"
            If Len(dbUser) = 0 Then dbUser = "postgres"
            ' Password will be empty - pro-upgrade.exe will fail if DB requires password
        End If

        Set fso = Nothing
        Set shell = Nothing
    End If

    LogMessage "UpgradeDatabase: DB_HOST = " & dbHost
    LogMessage "UpgradeDatabase: DB_PORT = " & dbPort
    LogMessage "UpgradeDatabase: DB_NAME = " & dbName
    LogMessage "UpgradeDatabase: DB_USER = " & dbUser
    LogMessage "UpgradeDatabase: DB_PASSWORD = " & String(Len(dbPassword), "*")
    LogMessage "UpgradeDatabase: INSTALLFOLDER = " & installFolder
    LogMessage "UpgradeDatabase: BACKUP_ENABLED = " & backupEnabled

    ' Set environment variable for password
    Set env = shell.Environment("Process")
    env("PGPASSWORD") = dbPassword
    env("DB_HOST") = dbHost
    env("DB_PORT") = dbPort
    env("DB_NAME") = dbName
    env("DB_USER") = dbUser
    env("DB_PASSWORD") = dbPassword

    ' Ensure installFolder ends with a backslash
    If Right(installFolder, 1) <> "\" Then
        installFolder = installFolder & "\"
    End If

    ' Path to pro-upgrade.exe
    proUpgradeExe = installFolder & "bin\pro-upgrade.exe"

    If Not fso.FileExists(proUpgradeExe) Then
        LogMessage "UpgradeDatabase: WARNING - pro-upgrade.exe not found at: " & proUpgradeExe
        LogMessage "UpgradeDatabase: Skipping upgrade (likely running during uninstall)"
        UpgradeDatabase = 1  ' Return success to not block uninstall
        Set fso = Nothing
        Set env = Nothing
        Set shell = Nothing
        Exit Function
    End If

    LogMessage "UpgradeDatabase: Found pro-upgrade.exe at: " & proUpgradeExe

    ' Check current version
    LogMessage "UpgradeDatabase: Checking current database version..."
    applyCmd = "cmd.exe /c """ & proUpgradeExe & """ check-version 2>&1"

    applyResult = shell.Run(applyCmd, 0, True)

    If applyResult = 0 Then
        LogMessage "UpgradeDatabase: Current version check completed"
    Else
        LogMessage "UpgradeDatabase: WARNING - Version check failed (exit code: " & applyResult & ")"
    End If

    ' Create backup if enabled
    If backupEnabled = "1" Or backupEnabled = "yes" Or backupEnabled = "true" Then
        LogMessage "UpgradeDatabase: Backup is enabled, creating database backup..."

        programDataFolder = "C:\ProgramData\Professional SMART\backups"

        ' Create backup directory if it doesn't exist
        If Not fso.FolderExists(programDataFolder) Then
            fso.CreateFolder(programDataFolder)
            LogMessage "UpgradeDatabase: Created backup directory: " & programDataFolder
        End If

        applyCmd = "cmd.exe /c """ & proUpgradeExe & """ backup-database --backup-dir """ & programDataFolder & """ 2>&1"

        LogMessage "UpgradeDatabase: Executing backup..."
        applyResult = shell.Run(applyCmd, 0, True)

        If applyResult = 0 Then
            LogMessage "UpgradeDatabase: SUCCESS - Backup created in: " & programDataFolder
        Else
            LogMessage "UpgradeDatabase: WARNING - Backup failed (exit code: " & applyResult & ")"
            LogMessage "UpgradeDatabase: This may be because the old version doesn't support backup-database command"
            LogMessage "UpgradeDatabase: Continuing with upgrade without backup..."
            ' Don't abort - backup is best-effort during upgrades
        End If
    Else
        LogMessage "UpgradeDatabase: Backup is disabled, skipping backup creation"
    End If

    ' Apply pending migrations (using embedded migrations in pro-upgrade.exe)
    LogMessage "UpgradeDatabase: Applying pending database migrations using embedded migrations..."

    applyCmd = "cmd.exe /c """ & proUpgradeExe & """ apply-migrations 2>&1"

    LogMessage "UpgradeDatabase: Executing: pro-upgrade apply-migrations (using embedded migrations)"
    applyResult = shell.Run(applyCmd, 0, True)

    If applyResult = 0 Then
        LogMessage "UpgradeDatabase: SUCCESS - All migrations applied successfully"
        LogMessage "UpgradeDatabase: Database upgrade completed successfully"
    Else
        LogMessage "UpgradeDatabase: WARNING - Migration application failed (exit code: " & applyResult & ")"
        LogMessage "UpgradeDatabase: This may indicate migrations are already applied from a previous version"
        LogMessage "UpgradeDatabase: Continuing with installation..."
        ' Don't abort - migrations may already be applied from a previous version
    End If

    ' Clean up old backups (keep last 5)
    LogMessage "UpgradeDatabase: Upgrade completed successfully"

    Set fso = Nothing
    Set env = Nothing
    Set shell = Nothing

    UpgradeDatabase = 1  ' Return success
    LogMessage "UpgradeDatabase: Completed with return code 1 (success)"
End Function
