' CheckPrerequisites.vbs - Check system prerequisites for Professional SMART

Function CheckPrerequisites()
    ' This function checks system prerequisites and sets properties
    ' for display in the prerequisite dialog

    Dim shell, fso
    Set shell = CreateObject("WScript.Shell")
    Set fso = CreateObject("Scripting.FileSystemObject")

    Dim allPassed
    allPassed = True

    ' Check 1: PostgreSQL (psql command)
    On Error Resume Next
    Dim psqlCheck
    psqlCheck = shell.Run("psql --version", 0, True)

    If psqlCheck = 0 Then
        Session.Property("PREREQ_PSQL_STATUS") = "PASSED"
    Else
        Session.Property("PREREQ_PSQL_STATUS") = "NOT FOUND"
        allPassed = False
    End If

    ' Check 2: PostgreSQL Service (check if any postgres process is running)
    Dim serviceCheck
    serviceCheck = shell.Run("powershell -Command ""Get-Service | Where-Object {$_.Name -like '*postgres*' -and $_.Status -eq 'Running'} | Select-Object -First 1""", 0, True)

    If serviceCheck = 0 Then
        Session.Property("PREREQ_PGSERVICE_STATUS") = "RUNNING"
    Else
        Session.Property("PREREQ_PGSERVICE_STATUS") = "NOT RUNNING"
        allPassed = False
    End If

    ' Check 3: Disk Space (check C: drive has at least 500 MB free)
    Dim drive
    Set drive = fso.GetDrive("C:")
    Dim freeSpaceMB
    freeSpaceMB = drive.FreeSpace / 1024 / 1024

    If freeSpaceMB >= 500 Then
        Session.Property("PREREQ_DISK_STATUS") = "OK"
    Else
        Session.Property("PREREQ_DISK_STATUS") = "LOW SPACE"
        allPassed = False
    End If

    On Error GoTo 0

    ' Set summary message
    If allPassed Then
        Session.Property("PREREQ_MESSAGE") = "All prerequisite checks passed. Click Next to continue."
    Else
        Session.Property("PREREQ_MESSAGE") = "Some checks failed. Installation will continue, but manual setup may be required."
    End If

    ' Always return success (don't block installation)
    CheckPrerequisites = 1

    Set fso = Nothing
    Set shell = Nothing
End Function
