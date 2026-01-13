//! MS SQL Server client using tiberius with SQL Server Authentication

use anyhow::{anyhow, Result};
use tiberius::{Client, Config};
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
        // Build ADO.NET style connection string with Encrypt=false and TrustServerCertificate=true
        let conn_str = format!(
            "Server={};Database={};User Id={};Password={};Encrypt=false;TrustServerCertificate=true",
            server,
            database,
            username.unwrap_or(""),
            password.unwrap_or("")
        );

        let config = Config::from_ado_string(&conn_str)
            .map_err(|e| anyhow!("Failed to parse connection string: {}", e))?;

        let addr = config.get_addr();
        let tcp = TcpStream::connect(&addr).await
            .map_err(|e| anyhow!("TCP connection to {} failed: {}. Verify SQL Server has TCP/IP enabled on port 1433.", addr, e))?;
        tcp.set_nodelay(true)?;

        let client = Client::connect(config, tcp.compat_write()).await
            .map_err(|e| anyhow!("SQL Server login failed: {}", e))?;

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
