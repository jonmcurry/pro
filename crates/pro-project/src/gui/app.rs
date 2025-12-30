extern crate native_windows_gui as nwg;
extern crate native_windows_derive as nwd;

use nwd::NwgUi;
use nwg::NativeUi;
use std::cell::RefCell;
use std::sync::mpsc;
use chrono::{DateTime, Utc};

// Base dimensions at 96 DPI (100% scaling)
// These are scaled at runtime based on actual DPI
const BASE_WINDOW_WIDTH: i32 = 960;
const BASE_WINDOW_HEIGHT: i32 = 680;
const BASE_MARGIN: i32 = 16;
const BASE_ROW_HEIGHT: i32 = 40;
const BASE_CONTROL_HEIGHT: i32 = 26;
const BASE_BTN_HEIGHT: i32 = 30;
const BASE_SECTION_GAP: i32 = 16;

// Color constants for status indicators (RGB)
const COLOR_SUCCESS: [u8; 3] = [34, 139, 34];    // Forest Green
const COLOR_WARNING: [u8; 3] = [184, 134, 11];   // Dark Goldenrod
const COLOR_ERROR: [u8; 3] = [178, 34, 34];      // Firebrick Red
const COLOR_INFO: [u8; 3] = [70, 130, 180];      // Steel Blue

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

/// Messages sent from background tasks to the GUI thread
#[derive(Debug)]
pub enum TaskMessage {
    Log(LogLevel, String),
    ProjectsLoaded(Result<Vec<ProjectRow>, String>),
    UpgradeProgress { database: String, migration: String, status: String },
    UpgradeComplete { succeeded: usize, failed: usize },
}

#[derive(Default, NwgUi)]
pub struct ProjectManagerApp {
    // Error state
    error: RefCell<Option<String>>,

    // Data
    projects: RefCell<Vec<ProjectRow>>,

    // Task communication
    task_receiver: RefCell<Option<mpsc::Receiver<TaskMessage>>>,

    // Font resources for modern appearance
    #[nwg_resource(family: "Segoe UI Semibold", size: 15)]
    header_font: nwg::Font,

    #[nwg_resource(family: "Segoe UI", size: 13)]
    body_font: nwg::Font,

    #[nwg_resource(family: "Consolas", size: 12)]
    log_font: nwg::Font,

    // Main Window
    #[nwg_control(size: (960, 680), position: (100, 100), title: "Professional SMART - Project Database Manager", flags: "WINDOW|VISIBLE|MINIMIZE_BOX|MAXIMIZE_BOX")]
    #[nwg_events(OnWindowClose: [ProjectManagerApp::exit], OnInit: [ProjectManagerApp::on_init])]
    window: nwg::Window,

    // Connection Section - Row 1 with proper spacing
    #[nwg_control(parent: window, text: "Host:", position: (12, 14), size: (40, 20))]
    lbl_host: nwg::Label,
    #[nwg_control(parent: window, text: "localhost", position: (55, 12), size: (100, 24))]
    txt_host: nwg::TextInput,

    #[nwg_control(parent: window, text: "Port:", position: (165, 14), size: (40, 20))]
    lbl_port: nwg::Label,
    #[nwg_control(parent: window, text: "5432", position: (208, 12), size: (55, 24))]
    txt_port: nwg::TextInput,

    #[nwg_control(parent: window, text: "User:", position: (273, 14), size: (40, 20))]
    lbl_user: nwg::Label,
    #[nwg_control(parent: window, text: "postgres", position: (316, 12), size: (85, 24))]
    txt_user: nwg::TextInput,

    #[nwg_control(parent: window, text: "Password:", position: (411, 14), size: (70, 20))]
    lbl_password: nwg::Label,
    #[nwg_control(parent: window, text: "", position: (484, 12), size: (100, 24), password: Some('*'))]
    txt_password: nwg::TextInput,

    #[nwg_control(parent: window, text: "Connect", position: (596, 10), size: (85, 28))]
    #[nwg_events(OnButtonClick: [ProjectManagerApp::connect])]
    btn_connect: nwg::Button,

    #[nwg_control(parent: window, text: "Refresh", position: (688, 10), size: (85, 28))]
    #[nwg_events(OnButtonClick: [ProjectManagerApp::refresh])]
    btn_refresh: nwg::Button,

    #[nwg_control(parent: window, position: (800, 12), size: (140, 24), flags: "VISIBLE")]
    rich_status: nwg::RichLabel,

    // Toolbar - Row 2
    #[nwg_control(parent: window, text: "Upgrade Selected", position: (12, 46), size: (130, 28))]
    #[nwg_events(OnButtonClick: [ProjectManagerApp::upgrade_selected])]
    btn_upgrade: nwg::Button,

    #[nwg_control(parent: window, text: "Backup && Upgrade", position: (150, 46), size: (140, 28))]
    #[nwg_events(OnButtonClick: [ProjectManagerApp::backup_and_upgrade])]
    btn_backup_upgrade: nwg::Button,

    #[nwg_control(parent: window, text: "Select All", position: (298, 46), size: (90, 28))]
    #[nwg_events(OnButtonClick: [ProjectManagerApp::select_all])]
    btn_select_all: nwg::Button,

    #[nwg_control(parent: window, text: "0 projects, 0 selected", position: (700, 50), size: (185, 20))]
    lbl_count: nwg::Label,

    // Projects ListView - full width with grid lines
    #[nwg_control(parent: window, position: (12, 82), size: (870, 290),
        list_style: nwg::ListViewStyle::Detailed,
        flags: "VISIBLE|TAB_STOP",
        focus: true,
        ex_flags: nwg::ListViewExFlags::FULL_ROW_SELECT | nwg::ListViewExFlags::GRID)]
    #[nwg_events(OnListViewItemChanged: [ProjectManagerApp::on_selection_changed])]
    projects_list: nwg::ListView,

    // Log Section - RichTextBox for colored log entries
    #[nwg_control(parent: window, text: "Activity Log", position: (16, 420), size: (100, 24))]
    log_label: nwg::Label,

    #[nwg_control(parent: window, position: (16, 448), size: (920, 200), flags: "VISIBLE|VSCROLL", readonly: true)]
    log_box: nwg::RichTextBox,

    // Timer for processing background messages
    #[nwg_control(interval: std::time::Duration::from_millis(100))]
    #[nwg_events(OnTimerTick: [ProjectManagerApp::process_messages])]
    timer: nwg::AnimationTimer,
}

impl ProjectManagerApp {
    pub fn get_error(&self) -> Option<String> {
        self.error.borrow().clone()
    }

    /// Set connection status with colored text
    fn set_connection_status(&self, text: &str, connected: bool) {
        let color = if connected { COLOR_SUCCESS } else { COLOR_WARNING };
        self.rich_status.set_text(text);
        let text_len = text.len() as u32;
        self.rich_status.set_char_format(0..text_len, &nwg::CharFormat {
            effects: Some(nwg::CharEffects::BOLD),
            height: None,
            y_offset: None,
            text_color: Some(color),
            font_face_name: Some("Segoe UI".to_string()),
            underline_type: None,
        });
    }

    fn on_init(&self) {
        // Apply DPI scaling first
        self.apply_dpi_scaling();

        // Apply fonts to controls
        self.log_label.set_font(Some(&self.header_font));
        self.log_box.set_font(Some(&self.log_font));

        // Initialize status
        self.set_connection_status("Not connected", false);

        // Set up ListView columns with scaled widths
        let scale = nwg::scale_factor() as f32;
        let s = |v: i32| -> i32 { (v as f32 * scale) as i32 };

        self.projects_list.insert_column(nwg::InsertListViewColumn {
            index: Some(0),
            fmt: None,
            width: Some(s(130)),
            text: Some("Database".to_string()),
        });
        self.projects_list.insert_column(nwg::InsertListViewColumn {
            index: Some(1),
            fmt: None,
            width: Some(s(150)),
            text: Some("Project".to_string()),
        });
        self.projects_list.insert_column(nwg::InsertListViewColumn {
            index: Some(2),
            fmt: None,
            width: Some(s(110)),
            text: Some("Organization".to_string()),
        });
        self.projects_list.insert_column(nwg::InsertListViewColumn {
            index: Some(3),
            fmt: None,
            width: Some(s(90)),
            text: Some("Version".to_string()),
        });
        self.projects_list.insert_column(nwg::InsertListViewColumn {
            index: Some(4),
            fmt: None,
            width: Some(s(100)),
            text: Some("Status".to_string()),
        });
        self.projects_list.insert_column(nwg::InsertListViewColumn {
            index: Some(5),
            fmt: None,
            width: Some(s(140)),
            text: Some("Last Used".to_string()),
        });
        self.projects_list.insert_column(nwg::InsertListViewColumn {
            index: Some(6),
            fmt: None,
            width: Some(s(70)),
            text: Some("Active".to_string()),
        });

        // Enable column headers - required for Detailed view to display properly
        self.projects_list.set_headers_enabled(true);

        // Try to load password from environment
        if let Ok(pwd) = std::env::var("DB_PASSWORD") {
            self.txt_password.set_text(&pwd);
        }

        self.add_log(LogLevel::Info, "Application started");
        self.add_log(LogLevel::Info, "Enter database credentials and click Connect");

        // Start timer for message processing
        self.timer.start();
    }

    fn apply_dpi_scaling(&self) {
        // Get the current scale factor (1.0 at 96 DPI, 1.25 at 120 DPI, etc.)
        let scale = nwg::scale_factor() as f32;

        // Helper to scale a value
        let s = |v: i32| -> i32 { (v as f32 * scale) as i32 };

        // Calculate scaled dimensions
        let margin = s(BASE_MARGIN);
        let row_height = s(BASE_ROW_HEIGHT);
        let control_height = s(BASE_CONTROL_HEIGHT);
        let btn_height = s(BASE_BTN_HEIGHT);
        let section_gap = s(BASE_SECTION_GAP);
        let window_width = s(BASE_WINDOW_WIDTH);
        let window_height = s(BASE_WINDOW_HEIGHT);
        let content_width = window_width - margin * 2;

        // Resize window
        self.window.set_size(window_width as u32, window_height as u32);

        // Row 1: Connection controls
        let mut y = margin;
        let mut x = margin;

        // Host
        self.lbl_host.set_position(x, y + 4);
        self.lbl_host.set_size(s(42) as u32, s(22) as u32);
        x += s(45);
        self.txt_host.set_position(x, y);
        self.txt_host.set_size(s(105) as u32, control_height as u32);
        x += s(113);

        // Port
        self.lbl_port.set_position(x, y + 4);
        self.lbl_port.set_size(s(38) as u32, s(22) as u32);
        x += s(40);
        self.txt_port.set_position(x, y);
        self.txt_port.set_size(s(58) as u32, control_height as u32);
        x += s(66);

        // User
        self.lbl_user.set_position(x, y + 4);
        self.lbl_user.set_size(s(38) as u32, s(22) as u32);
        x += s(40);
        self.txt_user.set_position(x, y);
        self.txt_user.set_size(s(90) as u32, control_height as u32);
        x += s(98);

        // Password
        self.lbl_password.set_position(x, y + 4);
        self.lbl_password.set_size(s(72) as u32, s(22) as u32);
        x += s(75);
        self.txt_password.set_position(x, y);
        self.txt_password.set_size(s(105) as u32, control_height as u32);
        x += s(113);

        // Connect button
        self.btn_connect.set_position(x, y - 2);
        self.btn_connect.set_size(s(88) as u32, btn_height as u32);
        x += s(95);

        // Refresh button
        self.btn_refresh.set_position(x, y - 2);
        self.btn_refresh.set_size(s(88) as u32, btn_height as u32);
        x += s(95);

        // Status label (RichLabel)
        self.rich_status.set_position(x, y + 2);
        self.rich_status.set_size(s(130) as u32, s(24) as u32);

        // Row 2: Toolbar
        y += row_height;
        x = margin;

        self.btn_upgrade.set_position(x, y);
        self.btn_upgrade.set_size(s(140) as u32, btn_height as u32);
        x += s(148);

        self.btn_backup_upgrade.set_position(x, y);
        self.btn_backup_upgrade.set_size(s(155) as u32, btn_height as u32);
        x += s(163);

        self.btn_select_all.set_position(x, y);
        self.btn_select_all.set_size(s(100) as u32, btn_height as u32);

        // Count label at right
        self.lbl_count.set_position(window_width - margin - s(210), y + 5);
        self.lbl_count.set_size(s(210) as u32, s(22) as u32);

        // Row 3: ListView
        y += row_height + section_gap / 2;
        let list_height = s(320);
        self.projects_list.set_position(margin, y);
        self.projects_list.set_size(content_width as u32, list_height as u32);

        // Row 4: Log section
        y += list_height + section_gap;
        self.log_label.set_position(margin, y);
        self.log_label.set_size(s(100) as u32, s(24) as u32);

        y += s(28);
        let log_height = window_height - y - margin;
        self.log_box.set_position(margin, y);
        self.log_box.set_size(content_width as u32, log_height as u32);
    }

    fn add_log(&self, level: LogLevel, message: &str) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        let level_str = match level {
            LogLevel::Info => "INFO",
            LogLevel::Success => "SUCCESS",
            LogLevel::Warning => "WARNING",
            LogLevel::Error => "ERROR",
        };

        // Get current text length to know where new text starts
        let current_text = self.log_box.text();
        let start_pos = current_text.len() as u32;

        // Format entry with Windows line ending
        let entry = format!("[{}] {} - {}\r\n", timestamp, level_str, message);

        // Append new entry
        self.log_box.set_text(&format!("{}{}", current_text, entry));

        // Color the level indicator
        let level_start = start_pos + timestamp.len() as u32 + 3; // After "[timestamp] "
        let level_end = level_start + level_str.len() as u32;

        let color = match level {
            LogLevel::Success => COLOR_SUCCESS,
            LogLevel::Warning => COLOR_WARNING,
            LogLevel::Error => COLOR_ERROR,
            LogLevel::Info => COLOR_INFO,
        };

        self.log_box.set_selection(level_start..level_end);
        self.log_box.set_char_format(&nwg::CharFormat {
            effects: Some(nwg::CharEffects::BOLD),
            height: None,
            y_offset: None,
            text_color: Some(color),
            font_face_name: None,
            underline_type: None,
        });

        // Scroll to end - move selection to end
        let total_len = self.log_box.text().len() as u32;
        self.log_box.set_selection(total_len..total_len);
    }

    fn connect(&self) {
        let host = self.txt_host.text();
        let port_str = self.txt_port.text();
        let user = self.txt_user.text();
        let password = self.txt_password.text();

        let port: u16 = port_str.parse().unwrap_or(5432);

        self.add_log(LogLevel::Info, &format!("Connecting to {}:{}...", host, port));
        self.set_connection_status("Connecting...", false);

        let (tx, rx) = mpsc::channel();
        *self.task_receiver.borrow_mut() = Some(rx);

        let host_clone = host.clone();
        let user_clone = user.clone();
        let password_clone = password.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                match crate::services::RegistryService::connect(&host_clone, port, &user_clone, &password_clone).await {
                    Ok(registry) => {
                        tx.send(TaskMessage::Log(LogLevel::Success, "Connected to SmartProAudit".to_string())).ok();

                        match registry.list_projects().await {
                            Ok(projects) => {
                                let migration_service = crate::services::MigrationService::new(&host_clone, port, &user_clone, &password_clone);

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

    fn refresh(&self) {
        self.projects_list.clear();
        *self.projects.borrow_mut() = Vec::new();
        self.connect();
    }

    fn update_projects_list(&self) {
        self.projects_list.clear();

        let projects = self.projects.borrow();
        for (idx, project) in projects.iter().enumerate() {
            self.projects_list.insert_item(nwg::InsertListViewItem {
                index: Some(idx as i32),
                column_index: 0,
                text: Some(project.database_name.clone()),
                image: None,
            });

            self.projects_list.insert_item(nwg::InsertListViewItem {
                index: Some(idx as i32),
                column_index: 1,
                text: Some(project.project_name.clone()),
                image: None,
            });

            self.projects_list.insert_item(nwg::InsertListViewItem {
                index: Some(idx as i32),
                column_index: 2,
                text: Some(project.organization.clone().unwrap_or_else(|| "-".to_string())),
                image: None,
            });

            self.projects_list.insert_item(nwg::InsertListViewItem {
                index: Some(idx as i32),
                column_index: 3,
                text: Some(project.database_version.clone().unwrap_or_else(|| "Unknown".to_string())),
                image: None,
            });

            self.projects_list.insert_item(nwg::InsertListViewItem {
                index: Some(idx as i32),
                column_index: 4,
                text: Some(project.status.to_string()),
                image: None,
            });

            self.projects_list.insert_item(nwg::InsertListViewItem {
                index: Some(idx as i32),
                column_index: 5,
                text: Some(
                    project.last_used_at
                        .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                        .unwrap_or_else(|| "-".to_string())
                ),
                image: None,
            });

            self.projects_list.insert_item(nwg::InsertListViewItem {
                index: Some(idx as i32),
                column_index: 6,
                text: Some(if project.is_active { "Active" } else { "-" }.to_string()),
                image: None,
            });
        }

        self.update_selection_count();
    }

    fn update_selection_count(&self) {
        let projects = self.projects.borrow();
        let total = projects.len();
        let selected = projects.iter().filter(|p| p.selected).count();
        self.lbl_count.set_text(&format!("{} projects, {} selected", total, selected));
    }

    fn on_selection_changed(&self) {
        // Update selected state based on ListView selection
        let selected_items = self.projects_list.selected_items();
        let mut projects = self.projects.borrow_mut();
        for (i, project) in projects.iter_mut().enumerate() {
            project.selected = selected_items.contains(&i);
        }
        drop(projects);
        self.update_selection_count();
    }

    fn select_all(&self) {
        let mut projects = self.projects.borrow_mut();
        let all_selected = projects.iter().all(|p| p.selected);
        for p in projects.iter_mut() {
            p.selected = !all_selected;
        }
        let new_state = !all_selected;
        let count = projects.len();
        drop(projects);

        // Update ListView selection
        for i in 0..count {
            self.projects_list.select_item(i, new_state);
        }

        self.update_selection_count();
    }

    fn upgrade_selected(&self) {
        self.do_upgrade(false);
    }

    fn backup_and_upgrade(&self) {
        self.do_upgrade(true);
    }

    fn do_upgrade(&self, with_backup: bool) {
        let selected: Vec<String> = {
            let projects = self.projects.borrow();
            projects.iter()
                .filter(|p| p.selected && matches!(p.status, ProjectStatus::PendingUpgrade(_)))
                .map(|p| p.database_name.clone())
                .collect()
        };

        if selected.is_empty() {
            self.add_log(LogLevel::Warning, "No databases selected for upgrade");
            nwg::simple_message("No Selection", "Please select databases with pending upgrades.");
            return;
        }

        let msg = if with_backup {
            format!("Backup and upgrade {} database(s)?", selected.len())
        } else {
            format!("Upgrade {} database(s) without backup?", selected.len())
        };

        if nwg::modal_message(&self.window, &nwg::MessageParams {
            title: "Confirm Upgrade",
            content: &msg,
            buttons: nwg::MessageButtons::YesNo,
            icons: nwg::MessageIcons::Question,
        }) != nwg::MessageChoice::Yes {
            return;
        }

        let host = self.txt_host.text();
        let port: u16 = self.txt_port.text().parse().unwrap_or(5432);
        let user = self.txt_user.text();
        let password = self.txt_password.text();

        let (tx, rx) = mpsc::channel();
        *self.task_receiver.borrow_mut() = Some(rx);

        self.add_log(LogLevel::Info, &format!("Starting upgrade of {} database(s)...", selected.len()));

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

    fn process_messages(&self) {
        let receiver = self.task_receiver.borrow();
        if let Some(ref rx) = *receiver {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    TaskMessage::Log(level, message) => {
                        self.add_log(level, &message);
                    }
                    TaskMessage::ProjectsLoaded(result) => {
                        match result {
                            Ok(projects) => {
                                let count = projects.len();
                                *self.projects.borrow_mut() = projects;
                                self.set_connection_status("Connected", true);
                                self.add_log(LogLevel::Success, &format!("Loaded {} projects", count));
                                drop(receiver);
                                self.update_projects_list();
                                return;
                            }
                            Err(e) => {
                                self.set_connection_status("Disconnected", false);
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
                            nwg::simple_message("Upgrade Complete", &msg);
                        } else {
                            self.add_log(LogLevel::Warning, &msg);
                            nwg::simple_message("Upgrade Complete", &msg);
                        }
                        drop(receiver);
                        self.refresh();
                        return;
                    }
                }
            }
        }
    }

    fn exit(&self) {
        self.timer.stop();
        nwg::stop_thread_dispatch();
    }
}
