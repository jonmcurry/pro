//! MS SQL Server client using tiberius with SQL Server Authentication

use anyhow::Result;
use tiberius::{Client, Config, AuthMethod};
use tokio::net::TcpStream;
use tokio_util::compat::TokioAsyncWriteCompatExt;

use crate::RuleRow;

pub struct MsSqlClient {
    client: Client<tokio_util::compat::Compat<TcpStream>>,
}

impl MsSqlClient {
    pub async fn connect_with_auth(
        server: &str,
        database: &str,
        username: Option<&str>,
        password: Option<&str>,
    ) -> Result<Self> {
        let mut config = Config::new();

        config.host(server);
        config.port(1433);
        config.database(database);

        // Use SQL Server Authentication
        if let (Some(user), Some(pass)) = (username, password) {
            config.authentication(AuthMethod::sql_server(user, pass));
        }

        // Note: TLS/encryption is disabled at compile time by not including
        // the rustls or native-tls features in Cargo.toml

        let tcp = TcpStream::connect(config.get_addr()).await?;
        tcp.set_nodelay(true)?;

        let client = Client::connect(config, tcp.compat_write()).await?;

        Ok(Self { client })
    }

    pub async fn query_rules(&mut self, sql: &str) -> Result<Vec<RuleRow>> {
        let stream = self.client.simple_query(sql).await?;
        let rows = stream.into_first_result().await?;

        let mut rules = Vec::new();

        for row in rows {
            let filter_number: &str = row.get(0).unwrap_or("");
            let filter_name: &str = row.get(1).unwrap_or("");
            let description: &str = row.get(2).unwrap_or("");
            let definition: &str = row.get(3).unwrap_or("");

            rules.push(RuleRow {
                filter_number: filter_number.to_string(),
                filter_name: filter_name.to_string(),
                description: description.to_string(),
                definition: definition.to_string(),
                selected: false,
            });
        }

        Ok(rules)
    }
}
