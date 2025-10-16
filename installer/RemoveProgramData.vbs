' Remove Professional SMART ProgramData folder
' This script removes the ProgramData folder silently without showing console windows

Function RemoveProgramData()
    On Error Resume Next

    Dim fso, shell, folder
    Set fso = CreateObject("Scripting.FileSystemObject")
    Set shell = CreateObject("WScript.Shell")

    ' Get the CommonAppDataFolder path from the custom action data
    ' The path is passed as the first parameter
    Dim programDataPath
    programDataPath = Session.Property("CustomActionData")

    ' Construct the full path
    folder = programDataPath & "Professional SMART"

    ' Remove the folder if it exists
    If fso.FolderExists(folder) Then
        fso.DeleteFolder folder, True
    End If

    RemoveProgramData = 0
End Function
