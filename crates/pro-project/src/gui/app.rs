use eframe::egui;
use tokio::sync::mpsc;
use chrono::{DateTime, Utc};

/// Project status for display in the GUI
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectStatus {
    UpToDate,
    PendingUpgrade(u32),
    /// Reserved for future error state display
    #[allow(dead_code)]
    Error(String),
    /// Reserved for future async checking state
    #[allow(dead_code)]
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
/// Fields are populated from database queries and used by the UI table renderer.
#[derive(Debug, Clone)]
#[allow(dead_code)]
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
pub struct LogEntry {
    pub level: LogLevel,
    pub message: String,
    pub timestamp: String,
}

impl LogEntry {
    fn new(level: LogLevel, message: String) -> Self {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        Self { level, message, timestamp }
    }
}

/// Messages sent from background tasks to the GUI thread
#[derive(Debug)]
#[allow(dead_code)]
pub enum TaskMessage {
    Log(LogLevel, String),
    ProjectsLoaded(Result<Vec<ProjectRow>, String>),
    UpgradeProgress { database: String, migration: String, status: String },
    UpgradeComplete { succeeded: usize, failed: usize },
}

pub struct ProjectManagerApp {
    // Connection settings
    db_host: String,
    db_port: String,
    db_user: String,
    db_password: String,
    connected: bool,
    connection_error: Option<String>,

    // Data
    projects: Vec<ProjectRow>,
    selected_count: usize,

    // UI state
    show_upgrade_dialog: bool,
    upgrade_in_progress: bool,
    upgrade_current_db: String,
    upgrade_current_migration: String,
    upgrade_progress: f32,
    upgrade_total: usize,
    upgrade_completed: usize,

    // Logs
    log_entries: Vec<LogEntry>,

    // Background task communication
    task_receiver: Option<mpsc::UnboundedReceiver<TaskMessage>>,
}

impl ProjectManagerApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self {
            db_host: "localhost".to_string(),
            db_port: "5432".to_string(),
            db_user: "postgres".to_string(),
            db_password: String::new(),
            connected: false,
            connection_error: None,
            projects: Vec::new(),
            selected_count: 0,
            show_upgrade_dialog: false,
            upgrade_in_progress: false,
            upgrade_current_db: String::new(),
            upgrade_current_migration: String::new(),
            upgrade_progress: 0.0,
            upgrade_total: 0,
            upgrade_completed: 0,
            log_entries: Vec::new(),
            task_receiver: None,
        };

        // Try to load password from environment
        if let Ok(pwd) = std::env::var("DB_PASSWORD") {
            app.db_password = pwd;
        }

        app.add_log(LogLevel::Info, "Application started".to_string());
        app.add_log(LogLevel::Info, "Enter database credentials and click Connect".to_string());

        app
    }

    fn add_log(&mut self, level: LogLevel, message: String) {
        self.log_entries.push(LogEntry::new(level, message));
        if self.log_entries.len() > 100 {
            self.log_entries.remove(0);
        }
    }

    fn connect(&mut self) {
        let host = self.db_host.clone();
        let port: u16 = self.db_port.parse().unwrap_or(5432);
        let user = self.db_user.clone();
        let password = self.db_password.clone();

        self.add_log(LogLevel::Info, format!("Connecting to {}:{}...", host, port));

        let (tx, rx) = mpsc::unbounded_channel();
        self.task_receiver = Some(rx);

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // Try to connect to SmartProAudit
                match crate::services::RegistryService::connect(&host, port, &user, &password).await {
                    Ok(registry) => {
                        tx.send(TaskMessage::Log(LogLevel::Success, "Connected to SmartProAudit".to_string())).ok();

                        // Load projects
                        match registry.list_projects().await {
                            Ok(projects) => {
                                let migration_service = crate::services::MigrationService::new(&host, port, &user, &password);

                                let mut rows = Vec::new();
                                for p in projects {
                                    // Get pending migrations
                                    let pending = migration_service
                                        .get_pending_migrations(&p.database_name)
                                        .await
                                        .unwrap_or_default();

                                    let status = if pending.is_empty() {
                                        ProjectStatus::UpToDate
                                    } else {
                                        ProjectStatus::PendingUpgrade(pending.len() as u32)
                                    };

                                    // Use database_version from SmartProAudit registry as source of truth
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

    fn refresh(&mut self) {
        self.connected = false;
        self.projects.clear();
        self.connect();
    }

    fn update_selection_count(&mut self) {
        self.selected_count = self.projects.iter().filter(|p| p.selected).count();
    }

    fn select_all(&mut self) {
        let all_selected = self.projects.iter().all(|p| p.selected);
        for p in &mut self.projects {
            p.selected = !all_selected;
        }
        self.update_selection_count();
    }

    fn upgrade_selected(&mut self, with_backup: bool) {
        let selected: Vec<_> = self.projects
            .iter()
            .filter(|p| p.selected && matches!(p.status, ProjectStatus::PendingUpgrade(_)))
            .map(|p| p.database_name.clone())
            .collect();

        if selected.is_empty() {
            self.add_log(LogLevel::Warning, "No databases selected for upgrade".to_string());
            return;
        }

        self.show_upgrade_dialog = true;
        self.upgrade_in_progress = true;
        self.upgrade_total = selected.len();
        self.upgrade_completed = 0;
        self.upgrade_progress = 0.0;

        let host = self.db_host.clone();
        let port: u16 = self.db_port.parse().unwrap_or(5432);
        let user = self.db_user.clone();
        let password = self.db_password.clone();

        let (tx, rx) = mpsc::unbounded_channel();
        self.task_receiver = Some(rx);

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

                    // Backup if requested
                    if with_backup {
                        tx.send(TaskMessage::Log(LogLevel::Info, format!("Creating backup of {}...", db_name))).ok();
                        if let Err(e) = backup_service.backup(db_name, None) {
                            tx.send(TaskMessage::Log(LogLevel::Error, format!("Backup failed: {}", e))).ok();
                            failed += 1;
                            continue;
                        }
                        tx.send(TaskMessage::Log(LogLevel::Success, format!("Backup created for {}", db_name))).ok();
                    }

                    // Get pending migrations
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

                        // Update database_version in SmartProAudit registry
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

    fn process_messages(&mut self, ctx: &egui::Context) {
        let messages: Vec<TaskMessage> = if let Some(receiver) = &mut self.task_receiver {
            let mut msgs = Vec::new();
            while let Ok(msg) = receiver.try_recv() {
                msgs.push(msg);
            }
            msgs
        } else {
            Vec::new()
        };

        for msg in messages {
            match msg {
                TaskMessage::Log(level, message) => {
                    self.add_log(level, message);
                }
                TaskMessage::ProjectsLoaded(result) => {
                    match result {
                        Ok(projects) => {
                            let count = projects.len();
                            self.projects = projects;
                            self.connected = true;
                            self.connection_error = None;
                            self.add_log(LogLevel::Success, format!("Loaded {} projects", count));
                        }
                        Err(e) => {
                            self.connection_error = Some(e.clone());
                            self.add_log(LogLevel::Error, format!("Failed to load projects: {}", e));
                        }
                    }
                }
                TaskMessage::UpgradeProgress { database, migration, status: _ } => {
                    self.upgrade_current_db = database;
                    self.upgrade_current_migration = migration;
                }
                TaskMessage::UpgradeComplete { succeeded, failed } => {
                    self.upgrade_in_progress = false;
                    self.add_log(
                        if failed == 0 { LogLevel::Success } else { LogLevel::Warning },
                        format!("Upgrade complete: {} succeeded, {} failed", succeeded, failed),
                    );
                    // Refresh project list
                    self.refresh();
                }
            }
        }

        if self.task_receiver.is_some() {
            ctx.request_repaint();
        }
    }
}

impl eframe::App for ProjectManagerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.process_messages(ctx);

        // Top panel - Header and connection
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.heading("Professional SMART - Project Database Manager");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.connected {
                        ui.colored_label(egui::Color32::GREEN, "● Connected");
                    } else if self.connection_error.is_some() {
                        ui.colored_label(egui::Color32::RED, "● Disconnected");
                    } else {
                        ui.label("● Not connected");
                    }
                });
            });
            ui.add_space(4.0);

            // Connection bar
            ui.horizontal(|ui| {
                ui.label("Host:");
                ui.add(egui::TextEdit::singleline(&mut self.db_host).desired_width(120.0));
                ui.label("Port:");
                ui.add(egui::TextEdit::singleline(&mut self.db_port).desired_width(50.0));
                ui.label("User:");
                ui.add(egui::TextEdit::singleline(&mut self.db_user).desired_width(80.0));
                ui.label("Password:");
                ui.add(egui::TextEdit::singleline(&mut self.db_password).password(true).desired_width(100.0));

                if ui.button("Connect").clicked() {
                    self.connect();
                }
                if self.connected && ui.button("Refresh").clicked() {
                    self.refresh();
                }
            });
            ui.add_space(4.0);
        });

        // Bottom panel - Log
        egui::TopBottomPanel::bottom("log").resizable(true).min_height(120.0).show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("Log");
            ui.separator();

            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                for entry in &self.log_entries {
                    ui.horizontal(|ui| {
                        ui.label(&entry.timestamp);
                        match entry.level {
                            LogLevel::Info => ui.label("INFO"),
                            LogLevel::Success => ui.colored_label(egui::Color32::GREEN, "SUCCESS"),
                            LogLevel::Warning => ui.colored_label(egui::Color32::from_rgb(255, 193, 7), "WARNING"),
                            LogLevel::Error => ui.colored_label(egui::Color32::RED, "ERROR"),
                        };
                        ui.label(&entry.message);
                    });
                }
            });
        });

        // Main content
        egui::CentralPanel::default().show(ctx, |ui| {
            if !self.connected {
                ui.vertical_centered(|ui| {
                    ui.add_space(100.0);
                    ui.heading("Connect to SmartProAudit database to view projects");
                    if let Some(ref error) = self.connection_error {
                        ui.add_space(20.0);
                        ui.colored_label(egui::Color32::RED, error);
                    }
                });
                return;
            }

            // Toolbar
            ui.horizontal(|ui| {
                let can_upgrade = self.projects.iter().any(|p| p.selected && matches!(p.status, ProjectStatus::PendingUpgrade(_)));

                if ui.add_enabled(can_upgrade, egui::Button::new("Upgrade Selected")).clicked() {
                    self.upgrade_selected(false);
                }
                if ui.add_enabled(can_upgrade, egui::Button::new("Backup & Upgrade")).clicked() {
                    self.upgrade_selected(true);
                }

                ui.separator();

                if ui.button("Select All").clicked() {
                    self.select_all();
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("{} projects, {} selected", self.projects.len(), self.selected_count));
                });
            });

            ui.add_space(8.0);

            // Project grid
            egui::ScrollArea::both().show(ui, |ui| {
                egui::Grid::new("project_grid")
                    .striped(true)
                    .min_col_width(50.0)
                    .show(ui, |ui| {
                        // Header row
                        ui.label("");
                        ui.strong("Database Name");
                        ui.strong("Project Name");
                        ui.strong("Organization");
                        ui.strong("Schema Version");
                        ui.strong("Status");
                        ui.strong("Last Used");
                        ui.strong("Active");
                        ui.end_row();

                        // Data rows
                        let mut selection_changed = false;
                        for project in &mut self.projects {
                            if ui.checkbox(&mut project.selected, "").changed() {
                                selection_changed = true;
                            }
                            ui.label(&project.database_name);
                            ui.label(&project.project_name);
                            ui.label(project.organization.as_deref().unwrap_or("-"));
                            ui.label(project.database_version.as_deref().unwrap_or("Unknown"));

                            // Status with color
                            match &project.status {
                                ProjectStatus::UpToDate => {
                                    ui.colored_label(egui::Color32::GREEN, "Up to date");
                                }
                                ProjectStatus::PendingUpgrade(n) => {
                                    ui.colored_label(egui::Color32::from_rgb(255, 193, 7), format!("{} pending", n));
                                }
                                ProjectStatus::Error(e) => {
                                    ui.colored_label(egui::Color32::RED, format!("Error: {}", e));
                                }
                                ProjectStatus::Checking => {
                                    ui.label("Checking...");
                                }
                            }

                            ui.label(
                                project.last_used_at
                                    .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                                    .unwrap_or_else(|| "-".to_string())
                            );

                            if project.is_active {
                                ui.colored_label(egui::Color32::GREEN, "★ Active");
                            } else {
                                ui.label("-");
                            }

                            ui.end_row();
                        }

                        if selection_changed {
                            self.update_selection_count();
                        }
                    });
            });
        });

        // Upgrade progress dialog
        if self.show_upgrade_dialog {
            egui::Window::new("Upgrading Databases")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.add_space(8.0);

                    if self.upgrade_in_progress {
                        ui.label(format!("Upgrading: {}", self.upgrade_current_db));
                        ui.label(format!("Migration: {}", self.upgrade_current_migration));
                        ui.add_space(8.0);
                        ui.add(egui::ProgressBar::new(self.upgrade_progress).show_percentage());
                        ui.add_space(8.0);
                        ui.add(egui::Spinner::new());
                    } else {
                        ui.label("Upgrade complete!");
                        ui.add_space(8.0);
                        if ui.button("Close").clicked() {
                            self.show_upgrade_dialog = false;
                        }
                    }
                });
        }
    }
}
