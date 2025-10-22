' TrimProperties.vbs - Trim whitespace from database properties

Function TrimDatabaseProperties()
    ' Trim all database-related properties
    Dim dbHost, dbPort, dbName, dbUser, dbPassword

    dbHost = Trim(Session.Property("DB_HOST"))
    dbPort = Trim(Session.Property("DB_PORT"))
    dbName = Trim(Session.Property("DB_NAME"))
    dbUser = Trim(Session.Property("DB_USER"))
    dbPassword = Trim(Session.Property("DB_PASSWORD"))

    ' Set them back
    Session.Property("DB_HOST") = dbHost
    Session.Property("DB_PORT") = dbPort
    Session.Property("DB_NAME") = dbName
    Session.Property("DB_USER") = dbUser
    Session.Property("DB_PASSWORD") = dbPassword

    TrimDatabaseProperties = 1
End Function
