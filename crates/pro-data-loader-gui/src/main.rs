//! Professional SMART - Master Data Loader GUI
//!
//! Modern egui-based GUI with eframe/glow backend.
//! Attempts software rendering when GPU is unavailable.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::mpsc;

use eframe::egui::{self, CentralPanel, RichText, Color32, Vec2, ProgressBar, ScrollArea};

use pro_data_loader::{Config, Facility, ImportResults, Organization, Provider, Region};

// Color constants for status indicators
const COLOR_SUCCESS: Color32 = Color32::from_rgb(34, 139, 34);    // Forest Green
const COLOR_WARNING: Color32 = Color32::from_rgb(184, 134, 11);   // Dark Goldenrod
const COLOR_ERROR: Color32 = Color32::from_rgb(178, 34, 34);      // Firebrick Red
const COLOR_INFO: Color32 = Color32::from_rgb(70, 130, 180);      // Steel Blue

#[derive(Debug, Clone, PartialEq)]
enum AppState {
    Idle,
    Validating,
    ValidationSuccess,
    ValidationError(String),
    Importing,
    ImportSuccess,
    ImportError(String),
}

impl Default for AppState {
    fn default() -> Self {
        AppState::Idle
    }
}

#[derive(Debug, Clone)]
enum LogLevel {
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

#[derive(Debug)]
struct ValidationData {
    organizations: Vec<Organization>,
    regions: Vec<Region>,
    facilities: Vec<Facility>,
    providers: Vec<Provider>,
}

#[derive(Debug)]
enum TaskMessage {
    Log(LogLevel, String),
    ValidationComplete(Result<ValidationData, String>),
    ImportProgress(f32),
    ImportComplete(Result<ImportResults, String>),
}

/// Application state
struct DataLoaderApp {
    // Configuration
    config: Option<Config>,
    app_state: AppState,
    validation_data: Option<ValidationData>,
    import_results: Option<ImportResults>,

    // File paths
    org_path: String,
    region_path: String,
    facility_path: String,
    provider_path: String,

    // Status
    db_status: String,
    db_connected: bool,
    status_message: String,
    status_level: LogLevel,
    progress: f32,

    // Log entries
    log_entries: Vec<LogEntry>,

    // Task communication
    task_receiver: Option<mpsc::Receiver<TaskMessage>>,
}

impl Default for DataLoaderApp {
    fn default() -> Self {
        Self {
            config: None,
            app_state: AppState::Idle,
            validation_data: None,
            import_results: None,
            org_path: String::new(),
            region_path: String::new(),
            facility_path: String::new(),
            provider_path: String::new(),
            db_status: "Loading configuration...".to_string(),
            db_connected: false,
            status_message: "Ready. Select CSV files and click Validate & Import.".to_string(),
            status_level: LogLevel::Info,
            progress: 0.0,
            log_entries: Vec::new(),
            task_receiver: None,
        }
    }
}

impl DataLoaderApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self::default();
        app.load_config();
        app.add_log(LogLevel::Info, "Application started");
        app
    }

    fn load_config(&mut self) {
        match Config::load() {
            Ok(config) => {
                self.db_status = format!("Database: {}", config.masked_url());
                self.db_connected = true;
                self.add_log(LogLevel::Success, "Database configuration loaded successfully");
                self.config = Some(config);
            }
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("Could not find .env file") {
                    self.db_status = "Database: Not configured".to_string();
                    self.add_log(LogLevel::Error, "Database configuration file not found");
                    self.add_log(LogLevel::Info, "Please run the Configuration Wizard");
                } else {
                    self.db_status = "Database: Configuration error".to_string();
                    self.add_log(LogLevel::Error, &format!("Config error: {}", error_msg));
                }
                self.db_connected = false;
            }
        }
    }

    fn add_log(&mut self, level: LogLevel, message: &str) {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        self.log_entries.push(LogEntry {
            timestamp,
            level,
            message: message.to_string(),
        });
    }

    fn set_status(&mut self, message: &str, level: LogLevel) {
        self.status_message = message.to_string();
        self.status_level = level;
    }

    fn process_messages(&mut self) {
        // Collect messages first to avoid borrow checker issues
        let messages: Vec<TaskMessage> = self.task_receiver
            .as_ref()
            .map(|rx| rx.try_iter().collect())
            .unwrap_or_default();

        // Now process all collected messages
        for msg in messages {
            match msg {
                TaskMessage::Log(level, message) => {
                    self.add_log(level, &message);
                }
                TaskMessage::ValidationComplete(result) => {
                    match result {
                        Ok(data) => {
                            let total = data.organizations.len() + data.regions.len()
                                + data.facilities.len() + data.providers.len();
                            self.app_state = AppState::ValidationSuccess;
                            self.set_status("Validation successful! Importing...", LogLevel::Success);
                            self.progress = 0.3;
                            self.add_log(LogLevel::Success, &format!("Validation successful! {} records to import", total));
                            self.validation_data = Some(data);
                        }
                        Err(e) => {
                            self.app_state = AppState::ValidationError(e.clone());
                            self.set_status(&format!("Validation failed: {}", e), LogLevel::Error);
                            self.progress = 0.0;
                        }
                    }
                }
                TaskMessage::ImportProgress(p) => {
                    self.progress = p;
                }
                TaskMessage::ImportComplete(result) => {
                    match result {
                        Ok(results) => {
                            self.app_state = AppState::ImportSuccess;
                            self.set_status("Import completed successfully!", LogLevel::Success);
                            self.progress = 1.0;
                            self.add_log(LogLevel::Success, &format!("Import completed! {} total records", results.total_inserted()));
                            self.import_results = Some(results);
                        }
                        Err(e) => {
                            self.app_state = AppState::ImportError(e.clone());
                            self.set_status(&format!("Import failed: {}", e), LogLevel::Error);
                            self.progress = 0.0;
                        }
                    }
                }
            }
        }
    }

    fn validate_and_import(&mut self) {
        if self.org_path.is_empty() || self.facility_path.is_empty() {
            self.add_log(LogLevel::Error, "Organizations and Facilities CSV files are required");
            return;
        }

        if self.config.is_none() {
            self.add_log(LogLevel::Error, "Database not configured");
            return;
        }

        self.app_state = AppState::Validating;
        self.set_status("Validating data...", LogLevel::Info);
        self.progress = 0.1;
        self.add_log(LogLevel::Info, "Starting validation...");

        let (tx, rx) = mpsc::channel();
        self.task_receiver = Some(rx);

        let org_path = self.org_path.clone();
        let region_path = self.region_path.clone();
        let facility_path = self.facility_path.clone();
        let provider_path = self.provider_path.clone();
        let database_url = self.config.as_ref().unwrap().database_url.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                tx.send(TaskMessage::Log(LogLevel::Info, "Parsing CSV files...".to_string())).ok();

                // Parse organizations
                let organizations = match pro_data_loader::csv_parser::parse_organizations(&org_path) {
                    Ok(o) => o,
                    Err(e) => {
                        tx.send(TaskMessage::ValidationComplete(Err(format!("Organizations: {}", e)))).ok();
                        return;
                    }
                };
                tx.send(TaskMessage::Log(LogLevel::Info, format!("Parsed {} organizations", organizations.len()))).ok();

                // Parse regions (optional)
                let regions = if !region_path.is_empty() {
                    match pro_data_loader::csv_parser::parse_regions(&region_path) {
                        Ok(r) => r,
                        Err(e) => {
                            tx.send(TaskMessage::ValidationComplete(Err(format!("Regions: {}", e)))).ok();
                            return;
                        }
                    }
                } else {
                    Vec::new()
                };
                if !regions.is_empty() {
                    tx.send(TaskMessage::Log(LogLevel::Info, format!("Parsed {} regions", regions.len()))).ok();
                }

                // Parse facilities
                let facilities = match pro_data_loader::csv_parser::parse_facilities(&facility_path) {
                    Ok(f) => f,
                    Err(e) => {
                        tx.send(TaskMessage::ValidationComplete(Err(format!("Facilities: {}", e)))).ok();
                        return;
                    }
                };
                tx.send(TaskMessage::Log(LogLevel::Info, format!("Parsed {} facilities", facilities.len()))).ok();

                // Parse providers (optional)
                let providers = if !provider_path.is_empty() {
                    match pro_data_loader::csv_parser::parse_providers(&provider_path) {
                        Ok(p) => p,
                        Err(e) => {
                            tx.send(TaskMessage::ValidationComplete(Err(format!("Providers: {}", e)))).ok();
                            return;
                        }
                    }
                } else {
                    Vec::new()
                };
                if !providers.is_empty() {
                    tx.send(TaskMessage::Log(LogLevel::Info, format!("Parsed {} providers", providers.len()))).ok();
                }

                // Validate
                tx.send(TaskMessage::Log(LogLevel::Info, "Validating data...".to_string())).ok();

                if let Err(e) = pro_data_loader::validator::validate_organizations(&organizations) {
                    tx.send(TaskMessage::ValidationComplete(Err(format!("Organizations validation: {}", e)))).ok();
                    return;
                }
                tx.send(TaskMessage::Log(LogLevel::Success, "Organizations: OK".to_string())).ok();

                if !regions.is_empty() {
                    if let Err(e) = pro_data_loader::validator::validate_regions(&regions, &organizations) {
                        tx.send(TaskMessage::ValidationComplete(Err(format!("Regions validation: {}", e)))).ok();
                        return;
                    }
                    tx.send(TaskMessage::Log(LogLevel::Success, "Regions: OK".to_string())).ok();
                }

                if let Err(e) = pro_data_loader::validator::validate_facilities(&facilities, &regions, &organizations) {
                    tx.send(TaskMessage::ValidationComplete(Err(format!("Facilities validation: {}", e)))).ok();
                    return;
                }
                tx.send(TaskMessage::Log(LogLevel::Success, "Facilities: OK".to_string())).ok();

                if !providers.is_empty() {
                    if let Err(e) = pro_data_loader::validator::validate_providers(&providers, &facilities) {
                        tx.send(TaskMessage::ValidationComplete(Err(format!("Providers validation: {}", e)))).ok();
                        return;
                    }
                    tx.send(TaskMessage::Log(LogLevel::Success, "Providers: OK".to_string())).ok();
                }

                tx.send(TaskMessage::ValidationComplete(Ok(ValidationData {
                    organizations: organizations.clone(),
                    regions: regions.clone(),
                    facilities: facilities.clone(),
                    providers: providers.clone(),
                }))).ok();

                // Start import
                tx.send(TaskMessage::Log(LogLevel::Info, "Starting import...".to_string())).ok();
                tx.send(TaskMessage::ImportProgress(0.2)).ok();

                let tx_clone = tx.clone();
                let result = pro_data_loader::importer::import_all_with_progress(
                    &database_url,
                    organizations,
                    regions,
                    facilities,
                    providers,
                    |msg| {
                        tx_clone.send(TaskMessage::Log(LogLevel::Info, msg.clone())).ok();
                        let progress = if msg.contains("organizations") { 0.4 }
                            else if msg.contains("regions") { 0.6 }
                            else if msg.contains("facilities") { 0.7 }
                            else if msg.contains("providers") { 0.9 }
                            else { 0.5 };
                        tx_clone.send(TaskMessage::ImportProgress(progress)).ok();
                    },
                ).await;

                match result {
                    Ok(results) => {
                        tx.send(TaskMessage::Log(LogLevel::Success, format!("Imported {} records", results.total_inserted()))).ok();
                        tx.send(TaskMessage::ImportComplete(Ok(results))).ok();
                    }
                    Err(e) => {
                        tx.send(TaskMessage::Log(LogLevel::Error, format!("Import failed: {}", e))).ok();
                        tx.send(TaskMessage::ImportComplete(Err(e.to_string()))).ok();
                    }
                }
            });
        });
    }
}

impl eframe::App for DataLoaderApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Process any pending background task messages
        self.process_messages();

        CentralPanel::default().show(ctx, |ui| {
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 12.0);

            // Title
            ui.heading(RichText::new("Professional SMART - Master Data Loader").size(24.0).strong());
            ui.add_space(8.0);

            // Database status
            let db_color = if self.db_connected { COLOR_SUCCESS } else { COLOR_WARNING };
            ui.label(RichText::new(&self.db_status).size(14.0).color(db_color).strong());
            ui.add_space(16.0);

            ui.separator();
            ui.add_space(8.0);

            // CSV File Selection section
            ui.label(RichText::new("CSV File Selection").size(16.0).strong());
            ui.add_space(8.0);

            // File input grid
            egui::Grid::new("file_inputs")
                .num_columns(3)
                .spacing([10.0, 8.0])
                .show(ui, |ui| {
                    // Organizations
                    ui.label("Organizations:");
                    ui.add(egui::TextEdit::singleline(&mut self.org_path).desired_width(400.0));
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("CSV", &["csv"])
                            .pick_file() {
                            self.org_path = path.display().to_string();
                            self.add_log(LogLevel::Info, &format!("Selected organizations: {}", self.org_path));
                        }
                    }
                    ui.end_row();

                    // Regions
                    ui.label("Regions (Optional):");
                    ui.add(egui::TextEdit::singleline(&mut self.region_path).desired_width(400.0));
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("CSV", &["csv"])
                            .pick_file() {
                            self.region_path = path.display().to_string();
                            self.add_log(LogLevel::Info, &format!("Selected regions: {}", self.region_path));
                        }
                    }
                    ui.end_row();

                    // Facilities
                    ui.label("Facilities:");
                    ui.add(egui::TextEdit::singleline(&mut self.facility_path).desired_width(400.0));
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("CSV", &["csv"])
                            .pick_file() {
                            self.facility_path = path.display().to_string();
                            self.add_log(LogLevel::Info, &format!("Selected facilities: {}", self.facility_path));
                        }
                    }
                    ui.end_row();

                    // Providers
                    ui.label("Providers (Optional):");
                    ui.add(egui::TextEdit::singleline(&mut self.provider_path).desired_width(400.0));
                    if ui.button("Browse...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("CSV", &["csv"])
                            .pick_file() {
                            self.provider_path = path.display().to_string();
                            self.add_log(LogLevel::Info, &format!("Selected providers: {}", self.provider_path));
                        }
                    }
                    ui.end_row();
                });

            ui.add_space(12.0);

            // Action buttons
            ui.horizontal(|ui| {
                if ui.button(RichText::new("Load from Directory...").size(14.0)).clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        let org = path.join("organizations.csv");
                        let region = path.join("regions.csv");
                        let facility = path.join("facilities.csv");
                        let provider = path.join("providers.csv");

                        let mut loaded = 0;
                        if org.exists() {
                            self.org_path = org.display().to_string();
                            loaded += 1;
                        }
                        if region.exists() {
                            self.region_path = region.display().to_string();
                            loaded += 1;
                        }
                        if facility.exists() {
                            self.facility_path = facility.display().to_string();
                            loaded += 1;
                        }
                        if provider.exists() {
                            self.provider_path = provider.display().to_string();
                            loaded += 1;
                        }

                        if loaded > 0 {
                            self.add_log(LogLevel::Info, &format!("Loaded {} CSV file(s) from directory", loaded));
                        } else {
                            self.add_log(LogLevel::Warning, "No CSV files found in directory");
                        }
                    }
                }

                if ui.button(RichText::new("Generate Templates...").size(14.0)).clicked() {
                    if let Some(path) = rfd::FileDialog::new().pick_folder() {
                        match pro_data_loader::templates::generate_templates(&path) {
                            Ok(_) => {
                                self.add_log(LogLevel::Success, &format!("Templates generated in: {}", path.display()));
                            }
                            Err(e) => {
                                self.add_log(LogLevel::Error, &format!("Failed to generate templates: {}", e));
                            }
                        }
                    }
                }

                if ui.button(RichText::new("Clear All").size(14.0)).clicked() {
                    self.org_path.clear();
                    self.region_path.clear();
                    self.facility_path.clear();
                    self.provider_path.clear();
                    self.add_log(LogLevel::Info, "Cleared all file selections");
                }
            });

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);

            // Status section
            ui.label(RichText::new("Status").size(16.0).strong());
            ui.add_space(4.0);

            let status_color = match self.status_level {
                LogLevel::Success => COLOR_SUCCESS,
                LogLevel::Warning => COLOR_WARNING,
                LogLevel::Error => COLOR_ERROR,
                LogLevel::Info => COLOR_INFO,
            };
            ui.label(RichText::new(&self.status_message).size(14.0).color(status_color));

            ui.add_space(8.0);
            ui.add(ProgressBar::new(self.progress).show_percentage());
            ui.add_space(8.0);

            // Results display
            if let Some(ref results) = self.import_results {
                ui.group(|ui| {
                    ui.label(RichText::new("Import Results").size(14.0).strong().color(COLOR_SUCCESS));
                    ui.label(format!("Organizations: {}", results.organizations_inserted));
                    ui.label(format!("Regions: {}", results.regions_inserted));
                    ui.label(format!("Facilities: {}", results.facilities_inserted));
                    ui.label(format!("Providers: {}", results.providers_inserted));
                    ui.label(RichText::new(format!("Total: {}", results.total_inserted())).strong());
                });
            }

            ui.add_space(8.0);

            // Import action buttons
            ui.horizontal(|ui| {
                let can_import = matches!(self.app_state, AppState::Idle | AppState::ValidationError(_) | AppState::ImportError(_) | AppState::ImportSuccess);

                if ui.add_enabled(can_import, egui::Button::new(RichText::new("Validate & Import").size(14.0))).clicked() {
                    self.validate_and_import();
                }

                if ui.button(RichText::new("Import More Data").size(14.0)).clicked() {
                    self.org_path.clear();
                    self.region_path.clear();
                    self.facility_path.clear();
                    self.provider_path.clear();
                    self.app_state = AppState::Idle;
                    self.validation_data = None;
                    self.import_results = None;
                    self.set_status("Ready. Select CSV files and click Validate & Import.", LogLevel::Info);
                    self.progress = 0.0;
                }
            });

            ui.add_space(16.0);
            ui.separator();
            ui.add_space(8.0);

            // Activity Log section
            ui.label(RichText::new("Activity Log").size(16.0).strong());
            ui.add_space(4.0);

            ScrollArea::vertical()
                .max_height(200.0)
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
        if matches!(self.app_state, AppState::Validating | AppState::Importing) {
            ctx.request_repaint();
        }
    }
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([920.0, 720.0])
            .with_min_inner_size([800.0, 600.0]),
        // Try to use software rendering if GPU is not available
        hardware_acceleration: eframe::HardwareAcceleration::Off,
        ..Default::default()
    };

    eframe::run_native(
        "Professional SMART - Master Data Loader",
        options,
        Box::new(|cc| Ok(Box::new(DataLoaderApp::new(cc)))),
    )
}
