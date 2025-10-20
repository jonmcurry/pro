' LoadEnvCredentials.vbs - Load database credentials from existing .env file
' This script reads the .env configuration file and sets MSI properties
' Used during upgrades to avoid re-prompting for database credentials

' Helper function to write to MSI log
Sub LogMessage(message)
    Dim record
    Set record = Session.Installer.CreateRecord(1)
    record.StringData(0) = "Professional SMART Installer: [1]"
    record.StringData(1) = message
    Session.Message &H04000000, record
End Sub

Function LoadEnvCredentials()
    LogMessage "LoadEnvCredentials: Starting to load credentials from .env file"

    ' Get ProgramData path
    Dim programDataFolder
    programDataFolder = Session.Property("CommonAppDataFolder")

    If Right(programDataFolder, 1) <> "\" Then
        programDataFolder = programDataFolder & "\"
    End If

    ' Build path to config file
    Dim configPath
    configPath = programDataFolder & "Professional SMART\config\.env"

    LogMessage "LoadEnvCredentials: Looking for config at: " & configPath

    ' Check if file exists
    Dim fso
    Set fso = CreateObject("Scripting.FileSystemObject")

    If Not fso.FileExists(configPath) Then
        LogMessage "LoadEnvCredentials: Config file not found - will use dialog input"
        Session.Property("ENV_CREDENTIALS_LOADED") = "0"
        LoadEnvCredentials = 1
        Set fso = Nothing
        Exit Function
    End If

    LogMessage "LoadEnvCredentials: Config file found, loading credentials..."

    ' Read the .env file and parse credentials
    On Error Resume Next
    Dim configFile, line
    Set configFile = fso.OpenTextFile(configPath, 1)

    If Err.Number <> 0 Then
        LogMessage "LoadEnvCredentials: ERROR - Could not open config file: " & Err.Description
        Session.Property("ENV_CREDENTIALS_LOADED") = "0"
        LoadEnvCredentials = 1
        Set fso = Nothing
        Exit Function
    End If

    ' Parse each line
    Dim dbHost, dbPort, dbName, dbUser, dbPassword
    dbHost = ""
    dbPort = ""
    dbName = ""
    dbUser = ""
    dbPassword = ""

    Do Until configFile.AtEndOfStream
        line = Trim(configFile.ReadLine)

        ' Skip comments and empty lines
        If Len(line) > 0 And Left(line, 1) <> "#" Then
            If InStr(line, "=") > 0 Then
                Dim parts
                parts = Split(line, "=", 2)
                Dim key, value
                key = Trim(parts(0))
                value = Trim(parts(1))

                ' Remove quotes if present
                If Left(value, 1) = """" And Right(value, 1) = """" Then
                    value = Mid(value, 2, Len(value) - 2)
                End If

                ' Extract credentials
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

    configFile.Close
    On Error GoTo 0

    ' Validate that we got all required credentials
    If dbHost <> "" And dbPort <> "" And dbName <> "" And dbUser <> "" And dbPassword <> "" Then
        LogMessage "LoadEnvCredentials: Successfully loaded all credentials"
        LogMessage "LoadEnvCredentials: DB_HOST = " & dbHost
        LogMessage "LoadEnvCredentials: DB_PORT = " & dbPort
        LogMessage "LoadEnvCredentials: DB_NAME = " & dbName
        LogMessage "LoadEnvCredentials: DB_USER = " & dbUser
        LogMessage "LoadEnvCredentials: DB_PASSWORD = " & String(Len(dbPassword), "*")

        ' Set MSI properties
        Session.Property("DB_HOST") = dbHost
        Session.Property("DB_PORT") = dbPort
        Session.Property("DB_NAME") = dbName
        Session.Property("DB_USER") = dbUser
        Session.Property("DB_PASSWORD") = dbPassword
        Session.Property("ENV_CREDENTIALS_LOADED") = "1"

        LogMessage "LoadEnvCredentials: Properties set successfully"
    Else
        LogMessage "LoadEnvCredentials: WARNING - Incomplete credentials in .env file"
        LogMessage "LoadEnvCredentials: Missing fields - will use dialog input"
        Session.Property("ENV_CREDENTIALS_LOADED") = "0"
    End If

    Set fso = Nothing
    LoadEnvCredentials = 1

    LogMessage "LoadEnvCredentials: Completed"
End Function
