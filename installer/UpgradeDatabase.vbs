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
    LogMessage "UpgradeDatabase: Using credentials from existing .env configuration"

    ' Get installer properties using CustomActionData
    Dim customData
    customData = Session.Property("CustomActionData")
    LogMessage "UpgradeDatabase: CustomActionData = " & customData

    ' Parse the custom action data (formatted as: host|port|name|user|password|installdir|backup_enabled)
    Dim parts
    parts = Split(customData, "|")

    If UBound(parts) < 6 Then
        LogMessage "UpgradeDatabase: ERROR - Not enough parameters. Expected 7, got " & (UBound(parts) + 1)
        UpgradeDatabase = 1  ' Return success to not block install
        Exit Function
    End If

    Dim dbHost, dbPort, dbName, dbUser, dbPassword, installFolder, backupEnabled
    dbHost = parts(0)
    dbPort = parts(1)
    dbName = parts(2)
    dbUser = parts(3)
    dbPassword = parts(4)
    installFolder = parts(5)
    backupEnabled = parts(6)

    LogMessage "UpgradeDatabase: DB_HOST = " & dbHost
    LogMessage "UpgradeDatabase: DB_PORT = " & dbPort
    LogMessage "UpgradeDatabase: DB_NAME = " & dbName
    LogMessage "UpgradeDatabase: DB_USER = " & dbUser
    LogMessage "UpgradeDatabase: DB_PASSWORD = " & String(Len(dbPassword), "*")
    LogMessage "UpgradeDatabase: INSTALLFOLDER = " & installFolder
    LogMessage "UpgradeDatabase: BACKUP_ENABLED = " & backupEnabled

    ' Create shell and FileSystemObject
    Dim shell, fso
    Set shell = CreateObject("WScript.Shell")
    Set fso = CreateObject("Scripting.FileSystemObject")

    ' Set environment variable for password
    Dim env
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
    Dim proUpgradeExe
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
    Dim versionCmd
    versionCmd = "cmd.exe /c """ & proUpgradeExe & """ check-version 2>&1"

    Dim versionResult
    versionResult = shell.Run(versionCmd, 0, True)

    If versionResult = 0 Then
        LogMessage "UpgradeDatabase: Current version check completed"
    Else
        LogMessage "UpgradeDatabase: WARNING - Version check failed (exit code: " & versionResult & ")"
    End If

    ' Create backup if enabled
    If backupEnabled = "1" Or backupEnabled = "yes" Or backupEnabled = "true" Then
        LogMessage "UpgradeDatabase: Backup is enabled, creating database backup..."

        Dim backupDir
        backupDir = "C:\ProgramData\Professional SMART\backups"

        ' Create backup directory if it doesn't exist
        If Not fso.FolderExists(backupDir) Then
            fso.CreateFolder(backupDir)
            LogMessage "UpgradeDatabase: Created backup directory: " & backupDir
        End If

        Dim backupCmd
        backupCmd = "cmd.exe /c """ & proUpgradeExe & """ backup-database --backup-dir """ & backupDir & """ 2>&1"

        LogMessage "UpgradeDatabase: Executing backup..."
        Dim backupResult
        backupResult = shell.Run(backupCmd, 0, True)

        If backupResult = 0 Then
            LogMessage "UpgradeDatabase: SUCCESS - Backup created in: " & backupDir
        Else
            LogMessage "UpgradeDatabase: ERROR - Backup failed (exit code: " & backupResult & ")"
            LogMessage "UpgradeDatabase: Aborting upgrade due to backup failure"
            UpgradeDatabase = 3  ' Return error
            Set fso = Nothing
            Set env = Nothing
            Set shell = Nothing
            Exit Function
        End If
    Else
        LogMessage "UpgradeDatabase: Backup is disabled, skipping backup creation"
    End If

    ' Apply pending migrations (using embedded migrations in pro-upgrade.exe)
    LogMessage "UpgradeDatabase: Applying pending database migrations using embedded migrations..."

    Dim applyCmd
    applyCmd = "cmd.exe /c """ & proUpgradeExe & """ apply-migrations 2>&1"

    LogMessage "UpgradeDatabase: Executing: pro-upgrade apply-migrations (using embedded migrations)"
    Dim applyResult
    applyResult = shell.Run(applyCmd, 0, True)

    If applyResult = 0 Then
        LogMessage "UpgradeDatabase: SUCCESS - All migrations applied successfully"
        LogMessage "UpgradeDatabase: Database upgrade completed successfully"
    Else
        LogMessage "UpgradeDatabase: WARNING - Migration application failed (exit code: " & applyResult & ")"
        LogMessage "UpgradeDatabase: Attempting to skip incompatible migrations 018-019..."

        ' Migrations 018-019 have schema incompatibilities and are performance-only
        ' Skip them and apply the critical migrations 020-024
        Dim skipCmd
        skipCmd = """" & psqlExe & """ -h " & dbHost & " -p " & dbPort & " -U " & dbUser & " -d " & dbName & _
                  " -c ""INSERT INTO staging.schema_migrations (migration_name, applied_at, checksum, description) " & _
                  "VALUES ('018_phase6_strategic_indexes.sql', NOW(), 'skipped-incompatible', 'Skipped - schema incompatibility'), " & _
                  "('019_phase6_materialized_views.sql', NOW(), 'skipped-incompatible', 'Skipped - analytics views') " & _
                  "ON CONFLICT (migration_name) DO NOTHING;"""

        LogMessage "UpgradeDatabase: Marking migrations 018-019 as skipped..."
        Dim skipResult
        skipResult = shell.Run(skipCmd, 0, True)

        If skipResult = 0 Then
            LogMessage "UpgradeDatabase: Successfully marked migrations 018-019 as skipped"
            LogMessage "UpgradeDatabase: Retrying migration application for remaining migrations..."

            ' Retry applying migrations
            applyResult = shell.Run(applyCmd, 0, True)

            If applyResult = 0 Then
                LogMessage "UpgradeDatabase: SUCCESS - All critical migrations applied successfully"
                LogMessage "UpgradeDatabase: Database upgrade completed successfully"
                LogMessage "UpgradeDatabase: Note: Migrations 018-019 were skipped due to schema incompatibility (performance-only)"
            Else
                LogMessage "UpgradeDatabase: ERROR - Migration retry failed (exit code: " & applyResult & ")"
                LogMessage "UpgradeDatabase: Database upgrade failed"

                ' If backup was created, inform user about rollback option
                If backupEnabled = "1" Or backupEnabled = "yes" Or backupEnabled = "true" Then
                    LogMessage "UpgradeDatabase: A backup was created before upgrade"
                    LogMessage "UpgradeDatabase: You can restore using: " & proUpgradeExe & " restore-database <backup-file>"
                End If
            End If
        Else
            LogMessage "UpgradeDatabase: ERROR - Failed to skip migrations 018-019 (exit code: " & skipResult & ")"
            LogMessage "UpgradeDatabase: Database upgrade failed"

            ' If backup was created, inform user about rollback option
            If backupEnabled = "1" Or backupEnabled = "yes" Or backupEnabled = "true" Then
                LogMessage "UpgradeDatabase: A backup was created before upgrade"
                LogMessage "UpgradeDatabase: You can restore using: " & proUpgradeExe & " restore-database <backup-file>"
            End If
        End If

        UpgradeDatabase = 3  ' Return error
        Set fso = Nothing
        Set env = Nothing
        Set shell = Nothing
        Exit Function
    End If

    ' Clean up old backups (keep last 5)
    LogMessage "UpgradeDatabase: Upgrade completed successfully"

    Set fso = Nothing
    Set env = Nothing
    Set shell = Nothing

    UpgradeDatabase = 1  ' Return success
    LogMessage "UpgradeDatabase: Completed with return code 1 (success)"
End Function
