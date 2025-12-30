//! Professional SMART - Project Database Manager GUI
//!
//! Modern egui-based GUI with eframe/wgpu backend.
//! Manages multiple project databases and their migrations.

use std::sync::mpsc;

use eframe::egui::{self, CentralPanel, RichText, Color32, Vec2, ScrollArea};
use chrono::{DateTime, Utc};

// Color constants for status indicators
const COLOR_SUCCESS: Color32 = Color32::from_rgb(34, 139, 34);    // Forest Green
const COLOR_WARNING: Color32 = Color32::from_rgb(184, 134, 11);   // Dark Goldenrod
const COLOR_ERROR: Color32 = Color32::from_rgb(178, 34, 34);      // Firebrick Red
const COLOR_INFO: Color32 = Color32::from_rgb(70, 130, 180);      // Steel Blue

/// Project status for display in the GUI
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectStatus {
    UpToDate,
    PendingUpgrade(u32),
    Error(String),
    Checking,
}

impl std::fmt::Display for ProjectStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProjectStatus::UpToDate => write!(f, "Up to date"),
            ProjectStatus::PendingUpgrade(n) => write!(f, "{} pending", n),
            ProjectStatus::Error(e) => write!(f, "Error: {}", e),
            ProjectStatus::Checking => write!(f, "Checking..."),
        }
    }
}

/// Data model for project rows displayed in the GUI data grid.
#[derive(Debug, Clone)]
pub struct ProjectRow {
    pub id: i32,
    pub project_name: String,
    pub database_name: String,
    pub organization: Option<String>,
    pub database_version: Option<String>,
    pub application_version: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub is_active: bool,
    pub notes: Option<String>,
    pub selected: bool,
    pub pending_migrations: u32,
    pub status: ProjectStatus,
}

#[derive(Debug, Clone)]
pub enum LogLevel {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
struct LogEntry {
    timestamp: String,
    level: LogLevel,
    message: String,
}

/// Messages sent from background tasks to the GUI thread
#[derive(Debug)]
pub enum TaskMessage {
    Log(LogLevel, String),
    ProjectsLoaded(Result<Vec<ProjectRow>, String>),
    UpgradeProgress { database: String, migration: String, status: String },
    UpgradeComplete { succeeded: usize, failed: usize },
}

#[derive(Debug, Clone, PartialEq)]
enum AppState {
    Disconnected,
    Connecting,
    Connected,
    Upgrading,
}

impl Default for AppState {
    fn default() -> Self {
        AppState::Disconnected
    }
}

/// Application state
pub struct ProjectManagerApp {
    // Connection settings
    host: String,
    port: String,
    user: String,
    password: String,

    // State
    app_state: AppState,
    projects: Vec<ProjectRow>,

    // Log entries
    log_entries: Vec<LogEntry>,

    // Task communication
    task_receiver: Option<mpsc::Receiver<TaskMessage>>,
}

impl Default for ProjectManagerApp {
    fn default() -> Self {
        // Try to load password from environment
        let password = std::env::var("DB_PASSWORD").unwrap_or_default();

        Self {
            host: "localhost".to_string(),
            port: "5432".to_string(),
            user: "postgres".to_string(),
            password,
            app_state: AppState::Disconnected,
            projects: Vec::new(),
            log_entries: Vec::new(),
            task_receiver: None,
        }
    }
}

impl ProjectManagerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::default();
        app.add_log(LogLevel::Info, "Application started");
        app.add_log(LogLevel::Info, "Enter database credentials and click Connect");
        app
    }

    fn add_log(&mut self, level: LogLevel, message: &str) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        self.log_entries.push(LogEntry {
            timestamp,
            level,
            message: message.to_string(),
        });
    }

    fn process_messages(&mut self) {
        // Collect messages first to avoid borrow checker issues
        let messages: Vec<TaskMessage> = self.task_receiver
            .as_ref()
            .map(|rx| rx.try_iter().collect())
            .unwrap_or_default();

        for msg in messages {
            match msg {
                TaskMessage::Log(level, message) => {
                    self.add_log(level, &message);
                }
                TaskMessage::ProjectsLoaded(result) => {
                    match result {
                        Ok(projects) => {
                            let count = projects.len();
                            self.projects = projects;
                            self.app_state = AppState::Connected;
                            self.add_log(LogLevel::Success, &format!("Loaded {} projects", count));
                        }
                        Err(e) => {
                            self.app_state = AppState::Disconnected;
                            self.add_log(LogLevel::Error, &format!("Failed to load projects: {}", e));
                        }
                    }
                }
                TaskMessage::UpgradeProgress { database, migration, status: _ } => {
                    self.add_log(LogLevel::Info, &format!("{}: {}", database, migration));
                }
                TaskMessage::UpgradeComplete { succeeded, failed } => {
                    let msg = format!("Upgrade complete: {} succeeded, {} failed", succeeded, failed);
                    if failed == 0 {
                        self.add_log(LogLevel::Success, &msg);
                    } else {
                        self.add_log(LogLevel::Warning, &msg);
                    }
                    self.app_state = AppState::Connected;
                    // Trigger a refresh
                    self.connect();
                }
            }
        }
    }

    fn connect(&mut self) {
        let port: u16 = self.port.parse().unwrap_or(5432);

        self.add_log(LogLevel::Info, &format!("Connecting to {}:{}...", self.host, port));
        self.app_state = AppState::Connecting;

        let (tx, rx) = mpsc::channel();
        self.task_receiver = Some(rx);

        let host = self.host.clone();
        let user = self.user.clone();
        let password = self.password.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                match crate::services::RegistryService::connect(&host, port, &user, &password).await {
                    Ok(registry) => {
                        tx.send(TaskMessage::Log(LogLevel::Success, "Connected to SmartProAudit".to_string())).ok();

                        match registry.list_projects().await {
                            Ok(projects) => {
                                let migration_service = crate::services::MigrationService::new(&host, port, &user, &password);

                                let mut rows = Vec::new();
                                for p in projects {
                                    let pending = migration_service
                                        .get_pending_migrations(&p.database_name)
                                        .await
                                        .unwrap_or_default();

                                    let status = if pending.is_empty() {
                                        ProjectStatus::UpToDate
                                    } else {
                                        ProjectStatus::PendingUpgrade(pending.len() as u32)
                                    };

                                    rows.push(ProjectRow {
                                        id: p.id,
                                        project_name: p.project_name,
                                        database_name: p.database_name,
                                        organization: p.organization,
                                        database_version: p.database_version,
                                        application_version: p.application_version,
                                        created_at: p.created_at,
                                        last_used_at: p.last_used_at,
                                        is_active: p.is_active,
                                        notes: p.notes,
                                        selected: false,
                                        pending_migrations: pending.len() as u32,
                                        status,
                                    });
                                }
                                tx.send(TaskMessage::ProjectsLoaded(Ok(rows))).ok();
                            }
                            Err(e) => {
                                tx.send(TaskMessage::ProjectsLoaded(Err(e.to_string()))).ok();
                            }
                        }
                    }
                    Err(e) => {
                        tx.send(TaskMessage::Log(LogLevel::Error, format!("Connection failed: {}", e))).ok();
                        tx.send(TaskMessage::ProjectsLoaded(Err(e.to_string()))).ok();
                    }
                }
            });
        });
    }

    fn do_upgrade(&mut self, with_backup: bool) {
        let selected: Vec<String> = self.projects.iter()
            .filter(|p| p.selected && matches!(p.status, ProjectStatus::PendingUpgrade(_)))
            .map(|p| p.database_name.clone())
            .collect();

        if selected.is_empty() {
            self.add_log(LogLevel::Warning, "No databases selected for upgrade");
            return;
        }

        self.app_state = AppState::Upgrading;
        self.add_log(LogLevel::Info, &format!("Starting upgrade of {} database(s)...", selected.len()));

        let (tx, rx) = mpsc::channel();
        self.task_receiver = Some(rx);

        let host = self.host.clone();
        let port: u16 = self.port.parse().unwrap_or(5432);
        let user = self.user.clone();
        let password = self.password.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let migration_service = crate::services::MigrationService::new(&host, port, &user, &password);
                let backup_service = crate::services::BackupService::new(
                    &host,
                    port,
                    &user,
                    &password,
                    crate::services::backup::default_backup_dir(),
                );
                let registry = crate::services::RegistryService::connect(&host, port, &user, &password).await.ok();

                let mut succeeded = 0;
                let mut failed = 0;

                for db_name in &selected {
                    tx.send(TaskMessage::UpgradeProgress {
                        database: db_name.clone(),
                        migration: "Starting...".to_string(),
                        status: "in_progress".to_string(),
                    }).ok();

                    if with_backup {
                        tx.send(TaskMessage::Log(LogLevel::Info, format!("Creating backup of {}...", db_name))).ok();
                        if let Err(e) = backup_service.backup(db_name, None) {
                            tx.send(TaskMessage::Log(LogLevel::Error, format!("Backup failed: {}", e))).ok();
                            failed += 1;
                            continue;
                        }
                        tx.send(TaskMessage::Log(LogLevel::Success, format!("Backup created for {}", db_name))).ok();
                    }

                    let pending = migration_service.get_pending_migrations(db_name).await.unwrap_or_default();
                    let all_migrations = crate::services::MigrationService::get_all_migrations();

                    let mut migration_failed = false;
                    for m in &pending {
                        tx.send(TaskMessage::UpgradeProgress {
                            database: db_name.clone(),
                            migration: m.file_name.clone(),
                            status: "applying".to_string(),
                        }).ok();

                        if let Some(migration) = all_migrations.iter().find(|em| em.version == m.version) {
                            match migration_service.apply_migration(db_name, migration).await {
                                Ok(()) => {
                                    tx.send(TaskMessage::Log(LogLevel::Success, format!("{}: Applied {}", db_name, m.file_name))).ok();
                                }
                                Err(e) => {
                                    tx.send(TaskMessage::Log(LogLevel::Error, format!("{}: Failed {}: {}", db_name, m.file_name, e))).ok();
                                    migration_failed = true;
                                    break;
                                }
                            }
                        }
                    }

                    if migration_failed {
                        failed += 1;
                    } else {
                        succeeded += 1;

                        if let Some(ref registry) = registry {
                            let applied = migration_service.get_applied_migrations(db_name).await.unwrap_or_default();
                            let new_version = if applied.is_empty() {
                                "Unknown".to_string()
                            } else {
                                let max_migration = applied.iter()
                                    .filter_map(|m| m.version.parse::<u32>().ok())
                                    .max()
                                    .unwrap_or(0);
                                format!("2.12.{}.0", max_migration)
                            };
                            if let Err(e) = registry.update_database_version(db_name, &new_version).await {
                                tx.send(TaskMessage::Log(LogLevel::Warning, format!("{}: Failed to update version: {}", db_name, e))).ok();
                            } else {
                                tx.send(TaskMessage::Log(LogLevel::Success, format!("{}: Updated version to {}", db_name, new_version))).ok();
                            }
                        }

                        tx.send(TaskMessage::Log(LogLevel::Success, format!("{}: Upgrade complete", db_name))).ok();
                    }
                }

                tx.send(TaskMessage::UpgradeComplete { succeeded, failed }).ok();
            });
        });
    }

    fn get_selected_count(&self) -> usize {
        self.projects.iter().filter(|p| p.selected).count()
    }

    fn get_selected_pending_count(&self) -> usize {
        self.projects.iter()
            .filter(|p| p.selected && matches!(p.status, ProjectStatus::PendingUpgrade(_)))
            .count()
    }

    fn select_all(&mut self) {
        let all_selected = self.projects.iter().all(|p| p.selected);
        for p in &mut self.projects {
            p.selected = !all_selected;
        }
    }
}

impl eframe::App for ProjectManagerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Process any pending background task messages
        self.process_messages();

        CentralPanel::default().show(ctx, |ui| {
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 12.0);

            // Title
            ui.heading(RichText::new("Professional SMART - Project Database Manager").size(24.0).strong());
            ui.add_space(8.0);

            ui.separator();
            ui.add_space(8.0);

            // Connection Section
            ui.label(RichText::new("Database Connection").size(16.0).strong());
            ui.add_space(4.0);

            ui.horizontal(|ui| {
                ui.label("Host:");
                ui.add(egui::TextEdit::singleline(&mut self.host).desired_width(120.0));
                ui.add_space(8.0);

                ui.label("Port:");
                ui.add(egui::TextEdit::singleline(&mut self.port).desired_width(60.0));
                ui.add_space(8.0);

                ui.label("User:");
                ui.add(egui::TextEdit::singleline(&mut self.user).desired_width(100.0));
                ui.add_space(8.0);

                ui.label("Password:");
                ui.add(egui::TextEdit::singleline(&mut self.password).password(true).desired_width(120.0));
                ui.add_space(16.0);

                let can_connect = !matches!(self.app_state, AppState::Connecting | AppState::Upgrading);
                if ui.add_enabled(can_connect, egui::Button::new(RichText::new("Connect").size(14.0))).clicked() {
                    self.connect();
                }

                if ui.add_enabled(can_connect && self.app_state == AppState::Connected,
                    egui::Button::new(RichText::new("Refresh").size(14.0))).clicked() {
                    self.projects.clear();
                    self.connect();
                }

                ui.add_space(16.0);

                // Connection status
                let (status_text, status_color) = match self.app_state {
                    AppState::Disconnected => ("Not connected", COLOR_WARNING),
                    AppState::Connecting => ("Connecting...", COLOR_INFO),
                    AppState::Connected => ("Connected", COLOR_SUCCESS),
                    AppState::Upgrading => ("Upgrading...", COLOR_INFO),
                };
                ui.label(RichText::new(status_text).size(14.0).color(status_color).strong());
            });

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);

            // Toolbar
            ui.horizontal(|ui| {
                let can_upgrade = self.app_state == AppState::Connected && self.get_selected_pending_count() > 0;

                if ui.add_enabled(can_upgrade, egui::Button::new(RichText::new("Upgrade Selected").size(14.0))).clicked() {
                    self.do_upgrade(false);
                }

                if ui.add_enabled(can_upgrade, egui::Button::new(RichText::new("Backup & Upgrade").size(14.0))).clicked() {
                    self.do_upgrade(true);
                }

                if ui.button(RichText::new("Select All").size(14.0)).clicked() {
                    self.select_all();
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let total = self.projects.len();
                    let selected = self.get_selected_count();
                    ui.label(RichText::new(format!("{} projects, {} selected", total, selected)).size(14.0));
                });
            });

            ui.add_space(8.0);

            // Projects Table
            ui.label(RichText::new("Project Databases").size(16.0).strong());
            ui.add_space(4.0);

            let available_height = ui.available_height() - 220.0; // Reserve space for log section

            ScrollArea::vertical()
                .max_height(available_height.max(150.0))
                .show(ui, |ui| {
                    egui::Grid::new("projects_grid")
                        .num_columns(8)
                        .spacing([10.0, 6.0])
                        .striped(true)
                        .min_col_width(60.0)
                        .show(ui, |ui| {
                            // Header row
                            ui.label(RichText::new("").strong()); // Checkbox column
                            ui.label(RichText::new("Database").strong());
                            ui.label(RichText::new("Project").strong());
                            ui.label(RichText::new("Organization").strong());
                            ui.label(RichText::new("Version").strong());
                            ui.label(RichText::new("Status").strong());
                            ui.label(RichText::new("Last Used").strong());
                            ui.label(RichText::new("Active").strong());
                            ui.end_row();

                            // Data rows
                            for project in &mut self.projects {
                                ui.checkbox(&mut project.selected, "");
                                ui.label(&project.database_name);
                                ui.label(&project.project_name);
                                ui.label(project.organization.as_deref().unwrap_or("-"));
                                ui.label(project.database_version.as_deref().unwrap_or("Unknown"));

                                // Status with color
                                let (status_text, status_color) = match &project.status {
                                    ProjectStatus::UpToDate => ("Up to date", COLOR_SUCCESS),
                                    ProjectStatus::PendingUpgrade(n) => {
                                        // Can't return formatted string, so handle inline
                                        let text = format!("{} pending", n);
                                        ui.label(RichText::new(text).color(COLOR_WARNING));
                                        ui.label(project.last_used_at
                                            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                                            .unwrap_or_else(|| "-".to_string()));
                                        ui.label(if project.is_active { "Active" } else { "-" });
                                        ui.end_row();
                                        continue;
                                    }
                                    ProjectStatus::Error(e) => {
                                        let text = format!("Error: {}", e);
                                        ui.label(RichText::new(text).color(COLOR_ERROR));
                                        ui.label(project.last_used_at
                                            .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                                            .unwrap_or_else(|| "-".to_string()));
                                        ui.label(if project.is_active { "Active" } else { "-" });
                                        ui.end_row();
                                        continue;
                                    }
                                    ProjectStatus::Checking => ("Checking...", COLOR_INFO),
                                };
                                ui.label(RichText::new(status_text).color(status_color));

                                ui.label(project.last_used_at
                                    .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                                    .unwrap_or_else(|| "-".to_string()));
                                ui.label(if project.is_active { "Active" } else { "-" });
                                ui.end_row();
                            }

                            // Show message if no projects
                            if self.projects.is_empty() {
                                ui.label("");
                                ui.label(RichText::new(
                                    if self.app_state == AppState::Connected {
                                        "No projects found"
                                    } else {
                                        "Connect to database to view projects"
                                    }
                                ).italics().color(Color32::GRAY));
                                ui.end_row();
                            }
                        });
                });

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);

            // Activity Log section
            ui.label(RichText::new("Activity Log").size(16.0).strong());
            ui.add_space(4.0);

            ScrollArea::vertical()
                .max_height(180.0)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for entry in &self.log_entries {
                        let level_color = match entry.level {
                            LogLevel::Success => COLOR_SUCCESS,
                            LogLevel::Warning => COLOR_WARNING,
                            LogLevel::Error => COLOR_ERROR,
                            LogLevel::Info => COLOR_INFO,
                        };
                        let level_str = match entry.level {
                            LogLevel::Info => "INFO",
                            LogLevel::Success => "SUCCESS",
                            LogLevel::Warning => "WARNING",
                            LogLevel::Error => "ERROR",
                        };

                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&format!("[{}]", entry.timestamp)).monospace().size(12.0));
                            ui.label(RichText::new(level_str).monospace().size(12.0).color(level_color).strong());
                            ui.label(RichText::new(&format!("- {}", entry.message)).monospace().size(12.0));
                        });
                    }
                });
        });

        // Request continuous repaints while tasks are running
        if matches!(self.app_state, AppState::Connecting | AppState::Upgrading) {
            ctx.request_repaint();
        }
    }
}
