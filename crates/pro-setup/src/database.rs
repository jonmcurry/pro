//! Database connection testing and validation

use anyhow::{Context, Result};
use sqlx::postgres::PgPoolOptions;

/// Test database connection
pub async fn test_database_connection(database_url: &str) -> Result<()> {
    // Create connection pool
    let pool = PgPoolOptions::new()
        .max_connections(1)
        .connect(database_url)
        .await
        .context("Failed to connect to database")?;

    // Test query
    let row: (i32,) = sqlx::query_as("SELECT 1")
        .fetch_one(&pool)
        .await
        .context("Failed to execute test query")?;

    if row.0 != 1 {
        return Err(anyhow::anyhow!("Test query returned unexpected result"));
    }

    println!("Database connection successful!");

    // Check if schemas exist
    let schemas: Vec<(String,)> = sqlx::query_as(
        "SELECT schema_name FROM information_schema.schemata WHERE schema_name IN ('staging', 'claims', 'ml')"
    )
    .fetch_all(&pool)
    .await
    .context("Failed to query schemas")?;

    if schemas.is_empty() {
        println!("Warning: Database schemas not found. Please run migrations.");
    } else {
        println!("Found {} schema(s): {}", schemas.len(), schemas.iter().map(|(s,)| s.as_str()).collect::<Vec<_>>().join(", "));
    }

    pool.close().await;

    Ok(())
}
