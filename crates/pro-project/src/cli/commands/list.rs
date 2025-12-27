use crate::cli::Cli;
use crate::services::{ConfigService, DatabaseService, RegistryService};
use anyhow::Result;

pub async fn execute(cli: &Cli, format: &str) -> Result<()> {
    let password = cli.db_password.as_deref().unwrap_or("");

    // Connect to SmartProAudit registry
    let registry = RegistryService::connect(&cli.db_host, cli.db_port, &cli.db_user, password).await?;
    let db_service = DatabaseService::new(&cli.db_host, cli.db_port, &cli.db_user, password);

    // Get current active database from config
    let config = ConfigService::with_default_path();
    let current_db = config.get_current_database().ok().flatten();

    // Get all projects
    let projects = registry.list_projects().await?;

    if projects.is_empty() {
        println!("No project databases registered.");
        println!();
        println!("Use 'pro-project create --name <NAME>' to create a new project.");
        return Ok(());
    }

    match format {
        "json" => print_json(&projects, &current_db)?,
        "csv" => print_csv(&projects, &current_db)?,
        _ => print_table(&projects, &current_db, &db_service).await?,
    }

    Ok(())
}

async fn print_table(
    projects: &[crate::services::ProjectInfo],
    current_db: &Option<String>,
    db_service: &DatabaseService,
) -> Result<()> {
    println!("PROJECT DATABASES");
    println!("=================");
    println!();
    println!(
        "  {:1} {:24} {:10} {:16} {:22} {:>10}",
        "", "NAME", "STATUS", "SCHEMA VERSION", "LAST USED", "SIZE"
    );
    println!(
        "  {:1} {:24} {:10} {:16} {:22} {:>10}",
        "", "------------------------", "----------", "----------------", "----------------------", "----------"
    );

    let mut accessible = 0;
    let mut offline = 0;

    for project in projects {
        let is_current = current_db.as_ref().map(|c| c == &project.database_name).unwrap_or(false);
        let marker = if is_current { "*" } else { " " };

        // Check if database is accessible
        let status = if db_service.database_exists(&project.database_name).await.unwrap_or(false) {
            accessible += 1;
            if is_current { "Active" } else { "Ready" }
        } else {
            offline += 1;
            "Offline"
        };

        let version = project.database_version.as_deref().unwrap_or("Unknown");

        let last_used = project
            .last_used_at
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_else(|| "Never".to_string());

        // Get size if accessible
        let size = if status != "Offline" {
            db_service
                .get_database_size(&project.database_name)
                .await
                .map(format_size)
                .unwrap_or_else(|_| "--".to_string())
        } else {
            "--".to_string()
        };

        println!(
            "{} {:24} {:10} {:16} {:22} {:>10}",
            marker,
            truncate(&project.database_name, 24),
            status,
            version,
            last_used,
            size
        );
    }

    println!();
    println!("* = Currently active database");
    if offline > 0 {
        println!("Offline = Database exists in registry but connection failed");
    }
    println!();
    println!(
        "Total: {} projects ({} accessible, {} offline)",
        projects.len(),
        accessible,
        offline
    );

    Ok(())
}

fn print_json(projects: &[crate::services::ProjectInfo], current_db: &Option<String>) -> Result<()> {
    let output: Vec<serde_json::Value> = projects
        .iter()
        .map(|p| {
            let is_active = current_db.as_ref().map(|c| c == &p.database_name).unwrap_or(false);
            serde_json::json!({
                "name": p.database_name,
                "project_name": p.project_name,
                "status": if is_active { "active" } else { "ready" },
                "schema_version": p.database_version,
                "last_used": p.last_used_at,
                "created_at": p.created_at,
                "organization": p.organization,
                "host": "localhost",
                "port": 5432
            })
        })
        .collect();

    let json_output = serde_json::json!({
        "projects": output,
        "total": projects.len(),
    });

    println!("{}", serde_json::to_string_pretty(&json_output)?);
    Ok(())
}

fn print_csv(projects: &[crate::services::ProjectInfo], current_db: &Option<String>) -> Result<()> {
    println!("name,project_name,status,schema_version,last_used,created_at,organization");

    for p in projects {
        let is_active = current_db.as_ref().map(|c| c == &p.database_name).unwrap_or(false);
        let status = if is_active { "active" } else { "ready" };
        let last_used = p.last_used_at.map(|dt| dt.to_rfc3339()).unwrap_or_default();
        let version = p.database_version.as_deref().unwrap_or("");
        let org = p.organization.as_deref().unwrap_or("");

        println!(
            "{},{},{},{},{},{},{}",
            p.database_name,
            p.project_name,
            status,
            version,
            last_used,
            p.created_at.to_rfc3339(),
            org
        );
    }

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len - 3])
    } else {
        s.to_string()
    }
}

fn format_size(bytes: i64) -> String {
    const KB: i64 = 1024;
    const MB: i64 = KB * 1024;
    const GB: i64 = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.0} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}
