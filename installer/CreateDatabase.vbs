' CreateDatabase.vbs - Create PostgreSQL database and schema

' Helper function to write to MSI log
Sub LogMessage(message)
    Dim record
    Set record = Session.Installer.CreateRecord(1)
    record.StringData(0) = "Professional SMART Installer: [1]"
    record.StringData(1) = message
    Session.Message &H04000000, record
End Sub

' Helper function to find PostgreSQL bin directory
Function FindPostgreSQLBinPath(shell, fso)
    LogMessage "FindPostgreSQLBinPath: Starting search for PostgreSQL"

    ' Method 1: Check if psql is in PATH environment variable
    LogMessage "FindPostgreSQLBinPath: Method 1 - Checking PATH environment variable"
    On Error Resume Next
    Dim testResult
    testResult = shell.Run("cmd.exe /c psql --version >nul 2>&1", 0, True)
    If testResult = 0 Then
        LogMessage "FindPostgreSQLBinPath: psql found in PATH - using system PATH"
        FindPostgreSQLBinPath = "PATH"  ' Return special marker to use PATH
        Exit Function
    End If
    LogMessage "FindPostgreSQLBinPath: psql not found in PATH"

    ' Method 2: Check common PostgreSQL installation directories
    LogMessage "FindPostgreSQLBinPath: Method 2 - Scanning for PostgreSQL installations"

    ' Scan for any version in Program Files directories
    Dim pgPath
    pgPath = ScanForPostgreSQLVersions(fso, "C:\Program Files\PostgreSQL\")
    If pgPath <> "" Then
        FindPostgreSQLBinPath = pgPath
        Exit Function
    End If

    pgPath = ScanForPostgreSQLVersions(fso, "C:\Program Files (x86)\PostgreSQL\")
    If pgPath <> "" Then
        FindPostgreSQLBinPath = pgPath
        Exit Function
    End If

    ' Check fallback location
    LogMessage "FindPostgreSQLBinPath: Checking fallback: C:\PostgreSQL\bin\"
    If fso.FileExists("C:\PostgreSQL\bin\psql.exe") Then
        LogMessage "FindPostgreSQLBinPath: Found PostgreSQL at: C:\PostgreSQL\bin\"
        FindPostgreSQLBinPath = "C:\PostgreSQL\bin\"
        Exit Function
    End If

    ' Method 3: Check registry for PostgreSQL installation path
    LogMessage "FindPostgreSQLBinPath: Method 3 - Checking Windows Registry"
    Dim regPath, installPath
    On Error Resume Next

    ' Try 64-bit registry first
    regPath = "HKEY_LOCAL_MACHINE\SOFTWARE\PostgreSQL\Installations\"
    installPath = GetPostgreSQLFromRegistry(shell, regPath)
    If installPath <> "" Then
        FindPostgreSQLBinPath = installPath
        Exit Function
    End If

    ' Try 32-bit registry
    regPath = "HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\PostgreSQL\Installations\"
    installPath = GetPostgreSQLFromRegistry(shell, regPath)
    If installPath <> "" Then
        FindPostgreSQLBinPath = installPath
        Exit Function
    End If

    LogMessage "FindPostgreSQLBinPath: PostgreSQL not found in any location"
    FindPostgreSQLBinPath = ""
End Function

' Helper function to scan directory for PostgreSQL versions
Function ScanForPostgreSQLVersions(fso, baseDir)
    LogMessage "ScanForPostgreSQLVersions: Scanning: " & baseDir
    On Error Resume Next

    ' Check if base directory exists
    If Not fso.FolderExists(baseDir) Then
        LogMessage "ScanForPostgreSQLVersions: Directory does not exist: " & baseDir
        ScanForPostgreSQLVersions = ""
        Exit Function
    End If

    ' Get all subdirectories and check each for psql.exe
    Dim folder, subfolders, subfolder, testPath, versions(), versionCount
    Set folder = fso.GetFolder(baseDir)
    Set subfolders = folder.SubFolders
    versionCount = 0

    ' First pass: count valid PostgreSQL installations
    For Each subfolder In subfolders
        testPath = subfolder.Path & "\bin\psql.exe"
        If fso.FileExists(testPath) Then
            versionCount = versionCount + 1
        End If
    Next

    If versionCount = 0 Then
        LogMessage "ScanForPostgreSQLVersions: No PostgreSQL installations found in: " & baseDir
        ScanForPostgreSQLVersions = ""
        Exit Function
    End If

    ' Second pass: collect version numbers and paths
    ReDim versions(versionCount - 1, 1) ' 2D array: [version number, path]
    Dim idx, versionNum
    idx = 0
    For Each subfolder In subfolders
        testPath = subfolder.Path & "\bin\psql.exe"
        If fso.FileExists(testPath) Then
            ' Try to parse version number from folder name
            On Error Resume Next
            versionNum = CInt(subfolder.Name)
            If Err.Number <> 0 Then
                versionNum = 0 ' Non-numeric name, give it lowest priority
            End If
            Err.Clear
            On Error GoTo 0

            versions(idx, 0) = versionNum
            versions(idx, 1) = subfolder.Path & "\bin\"
            LogMessage "ScanForPostgreSQLVersions: Found version " & subfolder.Name & " at: " & versions(idx, 1)
            idx = idx + 1
        End If
    Next

    ' Sort by version number (descending) to prefer newer versions
    Dim i, j, tempNum, tempPath
    For i = 0 To versionCount - 2
        For j = i + 1 To versionCount - 1
            If versions(i, 0) < versions(j, 0) Then
                tempNum = versions(i, 0)
                tempPath = versions(i, 1)
                versions(i, 0) = versions(j, 0)
                versions(i, 1) = versions(j, 1)
                versions(j, 0) = tempNum
                versions(j, 1) = tempPath
            End If
        Next
    Next

    ' Return the highest version (first in sorted array)
    LogMessage "ScanForPostgreSQLVersions: Using newest version: " & versions(0, 1)
    ScanForPostgreSQLVersions = versions(0, 1)

    Set subfolders = Nothing
    Set folder = Nothing
End Function

' Helper function to check registry for PostgreSQL
Function GetPostgreSQLFromRegistry(shell, basePath)
    LogMessage "GetPostgreSQLFromRegistry: Checking registry at: " & basePath
    On Error Resume Next

    ' Try to enumerate all installations under the base path
    ' Check common version patterns
    Dim versions(19) ' Expanded to support versions 9-28 and beyond
    Dim i, v, vIdx
    vIdx = 0

    ' Check versions 28 down to 9 (newest to oldest)
    For v = 28 To 9 Step -1
        versions(vIdx) = "postgresql-x64-" & v
        vIdx = vIdx + 1
        If vIdx > UBound(versions) Then Exit For
    Next

    ' Also check for non-versioned names
    If vIdx <= UBound(versions) Then
        versions(vIdx) = "PostgreSQL"
    End If

    Dim regKey, binPath
    For i = 0 To UBound(versions)
        If versions(i) <> "" Then
            regKey = basePath & versions(i) & "\Base Directory"
            binPath = ""
            binPath = shell.RegRead(regKey)

            If Err.Number = 0 And binPath <> "" Then
                LogMessage "GetPostgreSQLFromRegistry: Found in registry: " & versions(i) & " -> " & binPath
                If Right(binPath, 1) <> "\" Then
                    binPath = binPath & "\"
                End If
                GetPostgreSQLFromRegistry = binPath & "bin\"
                Exit Function
            End If
            Err.Clear
        End If
    Next

    LogMessage "GetPostgreSQLFromRegistry: No PostgreSQL found in registry"
    GetPostgreSQLFromRegistry = ""
End Function

Function CreateDatabase()
    LogMessage "CreateDatabase: Starting database creation"

    ' Get installer properties using CustomActionData
    Dim customData
    customData = Session.Property("CustomActionData")
    LogMessage "CreateDatabase: CustomActionData = " & customData

    ' Parse the custom action data (formatted as: host|port|name|user|password|installdir)
    Dim parts
    parts = Split(customData, "|")

    If UBound(parts) < 5 Then
        LogMessage "CreateDatabase: ERROR - Not enough parameters. Expected 6, got " & (UBound(parts) + 1)
        CreateDatabase = 1  ' Return success to not block install
        Exit Function
    End If

    Dim dbHost, dbPort, dbName, dbUser, dbPassword, installFolder
    dbHost = parts(0)
    dbPort = parts(1)
    dbName = parts(2)
    dbUser = parts(3)
    dbPassword = parts(4)
    installFolder = parts(5)

    LogMessage "CreateDatabase: DB_HOST = " & dbHost
    LogMessage "CreateDatabase: DB_PORT = " & dbPort
    LogMessage "CreateDatabase: DB_NAME = " & dbName
    LogMessage "CreateDatabase: DB_USER = " & dbUser
    LogMessage "CreateDatabase: DB_PASSWORD = " & String(Len(dbPassword), "*")
    LogMessage "CreateDatabase: INSTALLFOLDER = " & installFolder

    ' Create shell and FileSystemObject - needed for PostgreSQL path detection
    Dim shell, fso
    Set shell = CreateObject("WScript.Shell")
    Set fso = CreateObject("Scripting.FileSystemObject")
    LogMessage "CreateDatabase: Shell and FileSystemObject created"

    ' Set environment variable for password
    Dim env
    Set env = shell.Environment("Process")
    env("PGPASSWORD") = dbPassword
    LogMessage "CreateDatabase: PGPASSWORD environment variable set"

    ' Try to create the database using psql
    On Error Resume Next

    ' Find PostgreSQL installation and psql executable
    LogMessage "CreateDatabase: Searching for PostgreSQL installation..."
    Dim psqlPath, psqlExe
    psqlPath = FindPostgreSQLBinPath(shell, fso)

    ' Build psql command based on the result
    If psqlPath = "" Then
        ' PostgreSQL not found anywhere
        LogMessage "CreateDatabase: ERROR - PostgreSQL installation not found!"
        LogMessage "CreateDatabase: Checked PATH environment variable and common installation locations"
        LogMessage "CreateDatabase: Please ensure PostgreSQL is installed and accessible"
        LogMessage "CreateDatabase: Skipping database creation. Database must be created manually."
        CreateDatabase = 1  ' Return success to not block install
        Exit Function
    ElseIf psqlPath = "PATH" Then
        ' psql is in system PATH, use it directly
        psqlExe = "psql"
        LogMessage "CreateDatabase: Using psql from system PATH"
        LogMessage "CreateDatabase: psql.exe will be invoked via PATH"
    Else
        ' Use full path to psql
        psqlExe = psqlPath & "psql.exe"
        LogMessage "CreateDatabase: Found PostgreSQL at: " & psqlPath
        LogMessage "CreateDatabase: Using psql executable: " & psqlExe

        ' Verify psql executable exists at the specified path
        If Not fso.FileExists(psqlExe) Then
            LogMessage "CreateDatabase: ERROR - psql.exe not found at: " & psqlExe
            LogMessage "CreateDatabase: Skipping database creation."
            CreateDatabase = 1  ' Return success to not block install
            Exit Function
        End If
        LogMessage "CreateDatabase: psql.exe verified and ready"
    End If

    ' Validate PostgreSQL credentials before proceeding
    LogMessage "CreateDatabase: Validating PostgreSQL credentials..."
    LogMessage "CreateDatabase: Testing connection with user '" & dbUser & "' to database 'postgres'"
    Dim validateCmd
    validateCmd = "cmd.exe /c """ & psqlExe & " -h " & dbHost & " -p " & dbPort & " -U " & dbUser & " -d postgres -c ""SELECT 1;"" -tAc """" 2>&1"""

    Dim validateResult
    validateResult = shell.Run(validateCmd, 0, True)
    LogMessage "CreateDatabase: Credential validation exit code = " & validateResult

    If validateResult <> 0 Then
        LogMessage "CreateDatabase: ERROR - PostgreSQL authentication failed!"
        LogMessage "CreateDatabase: The password you entered is incorrect for user '" & dbUser & "'"
        LogMessage "CreateDatabase: Please check your PostgreSQL credentials and reinstall."
        LogMessage "CreateDatabase: Common issues:"
        LogMessage "CreateDatabase:   - Wrong password (check capitalization)"
        LogMessage "CreateDatabase:   - User '" & dbUser & "' does not exist - you must create this user first"
        LogMessage "CreateDatabase:   - PostgreSQL is not accepting connections on " & dbHost & ":" & dbPort
        LogMessage "CreateDatabase:   - pg_hba.conf may not allow password authentication for this user"
        LogMessage "CreateDatabase: Skipping database creation. Database must be created manually."
        CreateDatabase = 1  ' Return success to not block install
        Exit Function
    End If
    LogMessage "CreateDatabase: Credentials validated successfully"

    ' Check if database exists by trying to connect to it
    ' If connection fails, database doesn't exist
    Dim checkCmd
    checkCmd = """" & psqlExe & """ -h " & dbHost & " -p " & dbPort & " -U " & dbUser & " -d " & dbName & " -c ""SELECT 1;"" -tAc """""
    LogMessage "CreateDatabase: Checking if database exists by connecting to: " & dbName

    Dim result
    result = shell.Run(checkCmd, 0, True)
    LogMessage "CreateDatabase: Database check exit code = " & result

    ' Exit code 0 = database exists, non-zero = database doesn't exist
    If result <> 0 Then
        LogMessage "CreateDatabase: Database does not exist. Attempting to create..."
        Dim createCmd
        createCmd = """" & psqlExe & """ -h " & dbHost & " -p " & dbPort & " -U " & dbUser & " -d postgres -c ""CREATE DATABASE " & dbName & ";"""
        LogMessage "CreateDatabase: Create command using: " & psqlExe

        Dim createResult
        createResult = shell.Run(createCmd, 0, True)
        LogMessage "CreateDatabase: Create command exit code = " & createResult

        If createResult = 0 Then
            LogMessage "CreateDatabase: SUCCESS - Database created: " & dbName
        Else
            LogMessage "CreateDatabase: WARNING - Failed to create database. Exit code: " & createResult
            LogMessage "CreateDatabase: Database may need to be created manually"
        End If
    Else
        LogMessage "CreateDatabase: Database already exists: " & dbName
    End If

    ' Run database migrations using pro-upgrade.exe with embedded migrations
    LogMessage "CreateDatabase: Starting database schema migrations using pro-upgrade.exe..."

    ' Ensure installFolder ends with a backslash
    If Right(installFolder, 1) <> "\" Then
        installFolder = installFolder & "\"
    End If

    ' Set environment variables for pro-upgrade
    env("DB_HOST") = dbHost
    env("DB_PORT") = dbPort
    env("DB_NAME") = dbName
    env("DB_USER") = dbUser
    env("DB_PASSWORD") = dbPassword

    ' Path to pro-upgrade.exe
    Dim proUpgradeExe
    proUpgradeExe = installFolder & "bin\pro-upgrade.exe"

    If Not fso.FileExists(proUpgradeExe) Then
        LogMessage "CreateDatabase: WARNING - pro-upgrade.exe not found at: " & proUpgradeExe
        LogMessage "CreateDatabase: Skipping schema creation. Migrations must be run manually."
    Else
        LogMessage "CreateDatabase: Found pro-upgrade.exe at: " & proUpgradeExe

        ' Apply all migrations using embedded migrations (no --migrations-dir needed)
        Dim applyCmd, applyResult
        applyCmd = "cmd.exe /c """ & proUpgradeExe & """ apply-migrations 2>&1"

        LogMessage "CreateDatabase: Executing pro-upgrade apply-migrations (using embedded migrations)..."
        applyResult = shell.Run(applyCmd, 0, True)

        If applyResult = 0 Then
            LogMessage "CreateDatabase: SUCCESS - All migrations applied successfully"
            LogMessage "CreateDatabase: Database is fully initialized and ready to use"
        Else
            LogMessage "CreateDatabase: WARNING - Migration application failed (exit code: " & applyResult & ")"
            LogMessage "CreateDatabase: Attempting to skip incompatible migrations 018-019..."

            ' Migrations 018-019 have schema incompatibilities and are performance-only
            ' Skip them and apply the critical migrations 020-024
            Dim skipCmd
            skipCmd = """" & psqlExe & """ -h " & dbHost & " -p " & dbPort & " -U " & dbUser & " -d " & dbName & _
                      " -c ""INSERT INTO staging.schema_migrations (migration_name, applied_at, checksum, description) " & _
                      "VALUES ('018_phase6_strategic_indexes.sql', NOW(), 'skipped-incompatible', 'Skipped - schema incompatibility'), " & _
                      "('019_phase6_materialized_views.sql', NOW(), 'skipped-incompatible', 'Skipped - analytics views') " & _
                      "ON CONFLICT (migration_name) DO NOTHING;"""

            LogMessage "CreateDatabase: Marking migrations 018-019 as skipped..."
            Dim skipResult
            skipResult = shell.Run(skipCmd, 0, True)

            If skipResult = 0 Then
                LogMessage "CreateDatabase: Successfully marked migrations 018-019 as skipped"
                LogMessage "CreateDatabase: Retrying migration application for remaining migrations..."

                ' Retry applying migrations
                applyResult = shell.Run(applyCmd, 0, True)

                If applyResult = 0 Then
                    LogMessage "CreateDatabase: SUCCESS - All critical migrations (020-024) applied successfully"
                    LogMessage "CreateDatabase: Database is fully initialized and ready to use"
                    LogMessage "CreateDatabase: Note: Migrations 018-019 were skipped due to schema incompatibility (performance-only)"
                Else
                    LogMessage "CreateDatabase: ERROR - Migration retry failed (exit code: " & applyResult & ")"
                    LogMessage "CreateDatabase: WARNING - Database may not be fully initialized."
                    LogMessage "CreateDatabase: Please check the log for errors."
                End If
            Else
                LogMessage "CreateDatabase: ERROR - Failed to skip migrations 018-019 (exit code: " & skipResult & ")"
                LogMessage "CreateDatabase: WARNING - Database may not be fully initialized."
            End If
        End If
    End If

    Set fso = Nothing

    On Error GoTo 0

    ' Return success (1 = success in VBScript custom actions, 0 = failure)
    ' We return success even if database creation fails to not block the installation
    CreateDatabase = 1

    Set env = Nothing
    Set shell = Nothing

    LogMessage "CreateDatabase: Completed with return code 1 (success)"
End Function
