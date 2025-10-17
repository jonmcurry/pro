use eframe::egui;
use pro_data_loader::{Config, Facility, ImportResults, Organization, Provider, Region};
use tokio::sync::mpsc;

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

#[derive(Debug, Clone)]
enum LogLevel {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone)]
struct LogEntry {
    level: LogLevel,
    message: String,
    timestamp: String,
}

impl LogEntry {
    fn new(level: LogLevel, message: String) -> Self {
        let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
        Self {
            level,
            message,
            timestamp,
        }
    }
}

pub struct DataLoaderApp {
    // Database connection
    config: Option<Config>,
    db_status: String,
    db_connected: bool,

    // File paths
    org_path: String,
    region_path: String,
    facility_path: String,
    provider_path: String,

    // State
    app_state: AppState,

    // Validation results
    org_count: usize,
    region_count: usize,
    facility_count: usize,
    provider_count: usize,

    // Import results
    import_results: Option<ImportResults>,
    import_progress: f32,

    // Log
    log_entries: Vec<LogEntry>,

    // Background task communication
    task_receiver: Option<mpsc::UnboundedReceiver<TaskMessage>>,
}

#[derive(Debug)]
enum TaskMessage {
    Log(LogLevel, String),
    ValidationComplete(Result<ValidationData, String>),
    ImportProgress(f32),
    ImportComplete(Result<ImportResults, String>),
}

#[derive(Debug)]
struct ValidationData {
    organizations: Vec<Organization>,
    regions: Vec<Region>,
    facilities: Vec<Facility>,
    providers: Vec<Provider>,
}

impl DataLoaderApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self {
            config: None,
            db_status: String::new(),
            db_connected: false,
            org_path: String::new(),
            region_path: String::new(),
            facility_path: String::new(),
            provider_path: String::new(),
            app_state: AppState::Idle,
            org_count: 0,
            region_count: 0,
            facility_count: 0,
            provider_count: 0,
            import_results: None,
            import_progress: 0.0,
            log_entries: Vec::new(),
            task_receiver: None,
        };

        // Try to load configuration
        app.load_config();
        app
    }

    fn load_config(&mut self) {
        match Config::load() {
            Ok(config) => {
                self.db_status = format!("Connected: {}", config.masked_url());
                self.db_connected = true;
                self.add_log(LogLevel::Success, "Database configuration loaded successfully".to_string());
                self.config = Some(config);
            }
            Err(e) => {
                let error_msg = e.to_string();

                // Provide helpful error message based on specific error
                if error_msg.contains("Could not find .env file") {
                    self.db_status = "Configuration not found".to_string();
                    self.add_log(LogLevel::Error, "Database configuration file not found".to_string());
                    self.add_log(LogLevel::Info, "The system has not been configured yet.".to_string());
                    self.add_log(LogLevel::Info, "Please run the installation and configuration wizard.".to_string());
                } else if error_msg.contains("DATABASE_URL not found") || error_msg.contains("DATABASE_URL variable is missing") {
                    self.db_status = "DATABASE_URL missing".to_string();
                    self.add_log(LogLevel::Error, "DATABASE_URL not found in .env file".to_string());

                    // Show file contents preview if available
                    if error_msg.contains("File contents preview") {
                        if let Some(preview_start) = error_msg.find("File contents preview") {
                            if let Some(preview_end) = error_msg[preview_start..].find("Please run") {
                                let preview = &error_msg[preview_start..preview_start + preview_end];
                                self.add_log(LogLevel::Info, preview.to_string());
                            }
                        }
                    }

                    self.add_log(LogLevel::Info, "The .env file exists but DATABASE_URL is not set.".to_string());
                    self.add_log(LogLevel::Info, "Please run the Configuration Wizard to set up the database connection.".to_string());
                } else if error_msg.contains("is empty") {
                    self.db_status = "Configuration file empty".to_string();
                    self.add_log(LogLevel::Error, "The .env file is empty".to_string());
                    self.add_log(LogLevel::Info, "Please run the Configuration Wizard to set up the database connection.".to_string());
                } else if error_msg.contains("DATABASE_URL is empty") {
                    self.db_status = "DATABASE_URL empty".to_string();
                    self.add_log(LogLevel::Error, "DATABASE_URL is empty in .env file".to_string());
                    self.add_log(LogLevel::Info, "Please run the Configuration Wizard to set up the database connection.".to_string());
                } else if error_msg.contains("Cannot read .env file") {
                    self.db_status = "Cannot read config file".to_string();
                    self.add_log(LogLevel::Error, "Cannot read .env file".to_string());
                    self.add_log(LogLevel::Info, "Possible causes:".to_string());
                    self.add_log(LogLevel::Info, "- File permissions issue".to_string());
                    self.add_log(LogLevel::Info, "- File is locked by another process".to_string());
                    self.add_log(LogLevel::Info, "- File encoding issues".to_string());
                } else {
                    self.db_status = "Configuration error".to_string();
                    self.add_log(LogLevel::Error, "Failed to load database configuration".to_string());

                    // Split error message into lines for better readability
                    for line in error_msg.lines().take(5) {
                        if !line.trim().is_empty() {
                            self.add_log(LogLevel::Info, line.trim().to_string());
                        }
                    }
                }

                self.db_connected = false;
            }
        }
    }

    fn add_log(&mut self, level: LogLevel, message: String) {
        self.log_entries.push(LogEntry::new(level, message));
        // Keep only last 100 entries
        if self.log_entries.len() > 100 {
            self.log_entries.remove(0);
        }
    }

    fn browse_file(&mut self, field: &str) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("CSV Files", &["csv"])
            .pick_file()
        {
            let path_str = path.to_string_lossy().to_string();
            match field {
                "organizations" => {
                    self.org_path = path_str.clone();
                    self.add_log(LogLevel::Info, format!("Selected organizations file: {}", path_str));
                }
                "regions" => {
                    self.region_path = path_str.clone();
                    self.add_log(LogLevel::Info, format!("Selected regions file: {}", path_str));
                }
                "facilities" => {
                    self.facility_path = path_str.clone();
                    self.add_log(LogLevel::Info, format!("Selected facilities file: {}", path_str));
                }
                "providers" => {
                    self.provider_path = path_str.clone();
                    self.add_log(LogLevel::Info, format!("Selected providers file: {}", path_str));
                }
                _ => {}
            }
        }
    }

    fn browse_directory(&mut self) {
        if let Some(path) = rfd::FileDialog::new().pick_folder() {
            let org = path.join("organizations.csv");
            let region = path.join("regions.csv");
            let facility = path.join("facilities.csv");
            let provider = path.join("providers.csv");

            let mut loaded_count = 0;

            if org.exists() {
                self.org_path = org.to_string_lossy().to_string();
                loaded_count += 1;
            }
            if region.exists() {
                self.region_path = region.to_string_lossy().to_string();
                loaded_count += 1;
            }
            if facility.exists() {
                self.facility_path = facility.to_string_lossy().to_string();
                loaded_count += 1;
            }
            if provider.exists() {
                self.provider_path = provider.to_string_lossy().to_string();
                loaded_count += 1;
            }

            if loaded_count > 0 {
                self.add_log(LogLevel::Info, format!("Loaded {} CSV file(s) from directory: {}", loaded_count, path.display()));

                // Report missing optional files
                let mut missing_optional = Vec::new();
                if !region.exists() {
                    missing_optional.push("regions");
                }
                if !provider.exists() {
                    missing_optional.push("providers");
                }

                if !missing_optional.is_empty() {
                    self.add_log(LogLevel::Info, format!("{} file(s) not found - will be skipped (optional)", missing_optional.join(" and ")));
                }
            } else {
                self.add_log(LogLevel::Warning, format!("No CSV files found in directory: {}", path.display()));
            }
        }
    }

    fn generate_templates(&mut self) {
        if let Some(path) = rfd::FileDialog::new().pick_folder() {
            match pro_data_loader::templates::generate_templates(&path) {
                Ok(_) => {
                    self.add_log(LogLevel::Success, format!("Templates generated in: {}", path.display()));
                }
                Err(e) => {
                    self.add_log(LogLevel::Error, format!("Failed to generate templates: {}", e));
                }
            }
        }
    }

    fn clear_files(&mut self) {
        self.org_path.clear();
        self.region_path.clear();
        self.facility_path.clear();
        self.provider_path.clear();
        self.add_log(LogLevel::Info, "Cleared all file selections".to_string());
    }

    fn validate_data(&mut self) {
        // Check required files (only organizations and facilities are required)
        if self.org_path.is_empty() || self.facility_path.is_empty() {
            self.add_log(LogLevel::Error, "Organizations and Facilities CSV files are required".to_string());
            self.add_log(LogLevel::Info, "Regions and Providers CSV files are optional".to_string());
            return;
        }

        self.app_state = AppState::Validating;

        // Build status message for optional files
        let mut skipped = Vec::new();
        if self.region_path.is_empty() {
            skipped.push("regions");
        }
        if self.provider_path.is_empty() {
            skipped.push("providers");
        }

        if skipped.is_empty() {
            self.add_log(LogLevel::Info, "Starting validation...".to_string());
        } else {
            self.add_log(LogLevel::Info, format!("Starting validation ({} will be skipped)...", skipped.join(" and ")));
        }

        let (tx, rx) = mpsc::unbounded_channel();
        self.task_receiver = Some(rx);

        let org_path = self.org_path.clone();
        let region_path = self.region_path.clone();
        let facility_path = self.facility_path.clone();
        let provider_path = self.provider_path.clone();

        std::thread::spawn(move || {
            // Wrap in catch_unwind to handle panics gracefully
            let result = std::panic::catch_unwind(|| {
                let rt = match tokio::runtime::Runtime::new() {
                    Ok(rt) => rt,
                    Err(e) => {
                        tx.send(TaskMessage::Log(
                            LogLevel::Error,
                            format!("Failed to create async runtime: {}", e),
                        )).ok();
                        return;
                    }
                };

                rt.block_on(async {
                    tx.send(TaskMessage::Log(
                        LogLevel::Info,
                        "Parsing CSV files...".to_string(),
                    ))
                    .ok();

                let result = async {
                    // Parse organizations (required)
                    let organizations = pro_data_loader::csv_parser::parse_organizations(&org_path)?;
                    tx.send(TaskMessage::Log(
                        LogLevel::Info,
                        format!("Parsed {} organizations", organizations.len()),
                    ))
                    .ok();

                    // Parse regions only if file is provided
                    let regions = if !region_path.is_empty() {
                        let parsed = pro_data_loader::csv_parser::parse_regions(&region_path)?;
                        tx.send(TaskMessage::Log(
                            LogLevel::Info,
                            format!("Parsed {} regions", parsed.len()),
                        ))
                        .ok();
                        parsed
                    } else {
                        tx.send(TaskMessage::Log(
                            LogLevel::Info,
                            "No regions file provided - skipping regions".to_string(),
                        ))
                        .ok();
                        Vec::new()
                    };

                    // Parse facilities (required)
                    let facilities = pro_data_loader::csv_parser::parse_facilities(&facility_path)?;
                    tx.send(TaskMessage::Log(
                        LogLevel::Info,
                        format!("Parsed {} facilities", facilities.len()),
                    ))
                    .ok();

                    // Parse providers only if file is provided
                    let providers = if !provider_path.is_empty() {
                        let parsed = pro_data_loader::csv_parser::parse_providers(&provider_path)?;
                        tx.send(TaskMessage::Log(
                            LogLevel::Info,
                            format!("Parsed {} providers", parsed.len()),
                        ))
                        .ok();
                        parsed
                    } else {
                        tx.send(TaskMessage::Log(
                            LogLevel::Info,
                            "No providers file provided - skipping providers".to_string(),
                        ))
                        .ok();
                        Vec::new()
                    };

                    tx.send(TaskMessage::Log(
                        LogLevel::Info,
                        "Validating data...".to_string(),
                    ))
                    .ok();

                    // Validate organizations (required)
                    pro_data_loader::validator::validate_organizations(&organizations)?;
                    tx.send(TaskMessage::Log(
                        LogLevel::Success,
                        "Organizations: OK".to_string(),
                    ))
                    .ok();

                    // Validate regions only if we have any
                    if !regions.is_empty() {
                        pro_data_loader::validator::validate_regions(&regions, &organizations)?;
                        tx.send(TaskMessage::Log(
                            LogLevel::Success,
                            "Regions: OK".to_string(),
                        ))
                        .ok();
                    }

                    // Validate facilities (required)
                    pro_data_loader::validator::validate_facilities(&facilities, &regions, &organizations)?;
                    tx.send(TaskMessage::Log(
                        LogLevel::Success,
                        "Facilities: OK".to_string(),
                    ))
                    .ok();

                    // Validate providers only if we have any
                    if !providers.is_empty() {
                        pro_data_loader::validator::validate_providers(&providers, &facilities)?;
                        tx.send(TaskMessage::Log(
                            LogLevel::Success,
                            "Providers: OK".to_string(),
                        ))
                        .ok();
                    }

                    Ok::<_, anyhow::Error>(ValidationData {
                        organizations,
                        regions,
                        facilities,
                        providers,
                    })
                }
                .await;

                match result {
                    Ok(data) => {
                        tx.send(TaskMessage::ValidationComplete(Ok(data))).ok();
                    }
                    Err(e) => {
                        tx.send(TaskMessage::ValidationComplete(Err(e.to_string()))).ok();
                    }
                }
            });
            });

            // Handle panic from catch_unwind
            if let Err(panic_err) = result {
                let panic_msg = if let Some(s) = panic_err.downcast_ref::<&str>() {
                    format!("Thread panicked: {}", s)
                } else if let Some(s) = panic_err.downcast_ref::<String>() {
                    format!("Thread panicked: {}", s)
                } else {
                    "Thread panicked with unknown error".to_string()
                };
                tx.send(TaskMessage::ValidationComplete(Err(panic_msg))).ok();
            }
        });
    }

    fn import_data(&mut self, data: ValidationData) {
        if let Some(ref config) = self.config {
            let database_url = config.database_url.clone();

            self.app_state = AppState::Importing;
            self.import_progress = 0.0;
            self.add_log(LogLevel::Info, "Starting import...".to_string());

            let (tx, rx) = mpsc::unbounded_channel();
            self.task_receiver = Some(rx);

            std::thread::spawn(move || {
                // Wrap in catch_unwind to handle panics gracefully
                let result = std::panic::catch_unwind(|| {
                    let rt = match tokio::runtime::Runtime::new() {
                        Ok(rt) => rt,
                        Err(e) => {
                            tx.send(TaskMessage::Log(
                                LogLevel::Error,
                                format!("Failed to create async runtime: {}", e),
                            )).ok();
                            tx.send(TaskMessage::ImportComplete(Err(format!("Failed to create async runtime: {}", e)))).ok();
                            return;
                        }
                    };

                    rt.block_on(async {
                        tx.send(TaskMessage::ImportProgress(0.1)).ok();
                        tx.send(TaskMessage::Log(
                            LogLevel::Info,
                            "Connecting to database...".to_string(),
                        ))
                        .ok();

                        let tx_clone = tx.clone();

                        let result = pro_data_loader::importer::import_all_with_progress(
                            &database_url,
                            data.organizations,
                            data.regions,
                            data.facilities,
                            data.providers,
                            |msg| {
                                tx_clone.send(TaskMessage::Log(LogLevel::Info, msg.clone())).ok();

                                // Update progress based on stage
                                let progress = if msg.contains("Connecting") {
                                    0.15
                                } else if msg.contains("Starting transaction") {
                                    0.25
                                } else if msg.contains("Importing") && msg.contains("organizations") {
                                    0.35
                                } else if msg.contains("Inserted") && msg.contains("organizations") {
                                    0.45
                                } else if msg.contains("Importing") && msg.contains("regions") {
                                    0.50
                                } else if msg.contains("Inserted") && msg.contains("regions") {
                                    0.60
                                } else if msg.contains("Importing") && msg.contains("facilities") {
                                    0.65
                                } else if msg.contains("Inserted") && msg.contains("facilities") {
                                    0.75
                                } else if msg.contains("Importing") && msg.contains("providers") {
                                    0.80
                                } else if msg.contains("Inserted") && msg.contains("providers") {
                                    0.90
                                } else if msg.contains("Committing") {
                                    0.95
                                } else if msg.contains("completed successfully") {
                                    1.0
                                } else {
                                    return; // Don't update progress for other messages
                                };

                                tx_clone.send(TaskMessage::ImportProgress(progress)).ok();
                            },
                        )
                        .await;

                        tx.send(TaskMessage::ImportProgress(1.0)).ok();

                        match result {
                            Ok(results) => {
                                tx.send(TaskMessage::Log(
                                    LogLevel::Success,
                                    format!("Successfully imported {} records", results.total_inserted()),
                                ))
                                .ok();
                                tx.send(TaskMessage::ImportComplete(Ok(results))).ok();
                            }
                            Err(e) => {
                                tx.send(TaskMessage::Log(
                                    LogLevel::Error,
                                    format!("Import failed: {}", e),
                                ))
                                .ok();
                                tx.send(TaskMessage::ImportComplete(Err(e.to_string()))).ok();
                            }
                        }
                    });
                });

                // Handle panic from catch_unwind
                if let Err(panic_err) = result {
                    let panic_msg = if let Some(s) = panic_err.downcast_ref::<&str>() {
                        format!("Import thread panicked: {}", s)
                    } else if let Some(s) = panic_err.downcast_ref::<String>() {
                        format!("Import thread panicked: {}", s)
                    } else {
                        "Import thread panicked with unknown error".to_string()
                    };
                    tx.send(TaskMessage::Log(LogLevel::Error, panic_msg.clone())).ok();
                    tx.send(TaskMessage::ImportComplete(Err(panic_msg))).ok();
                }
            });
        }
    }

    fn process_messages(&mut self, ctx: &egui::Context) {
        // Collect all messages first to avoid borrow checker issues
        let messages: Vec<TaskMessage> = if let Some(receiver) = &mut self.task_receiver {
            let mut msgs = Vec::new();
            while let Ok(msg) = receiver.try_recv() {
                msgs.push(msg);
            }
            msgs
        } else {
            Vec::new()
        };

        // Process collected messages
        for msg in messages {
            match msg {
                TaskMessage::Log(level, message) => {
                    self.add_log(level, message);
                }
                TaskMessage::ValidationComplete(result) => {
                    match result {
                        Ok(data) => {
                            self.org_count = data.organizations.len();
                            self.region_count = data.regions.len();
                            self.facility_count = data.facilities.len();
                            self.provider_count = data.providers.len();
                            self.app_state = AppState::ValidationSuccess;
                            self.add_log(
                                LogLevel::Success,
                                format!(
                                    "Validation successful! Ready to import {} total records",
                                    self.org_count + self.region_count + self.facility_count + self.provider_count
                                ),
                            );
                            // Clear the validation task receiver before starting import
                            self.task_receiver = None;
                            // Automatically start import after successful validation
                            self.import_data(data);
                        }
                        Err(e) => {
                            self.app_state = AppState::ValidationError(e.clone());
                            self.add_log(LogLevel::Error, format!("Validation failed: {}", e));
                            self.task_receiver = None;
                        }
                    }
                }
                TaskMessage::ImportProgress(progress) => {
                    self.import_progress = progress;
                }
                TaskMessage::ImportComplete(result) => {
                    match result {
                        Ok(results) => {
                            self.import_results = Some(results.clone());
                            self.app_state = AppState::ImportSuccess;
                            self.add_log(
                                LogLevel::Success,
                                format!(
                                    "Import completed! {} total records imported",
                                    results.total_inserted()
                                ),
                            );
                        }
                        Err(e) => {
                            self.app_state = AppState::ImportError(e.clone());
                            self.add_log(LogLevel::Error, format!("Import failed: {}", e));
                        }
                    }
                    self.task_receiver = None;
                }
            }
        }

        // Request repaint if we're in a busy state
        if matches!(
            self.app_state,
            AppState::Validating | AppState::Importing
        ) || self.task_receiver.is_some()
        {
            ctx.request_repaint();
        }
    }
}

impl eframe::App for DataLoaderApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Process background task messages
        self.process_messages(ctx);

        // Top panel - Header
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.heading("Professional SMART - Master Data Loader");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.db_connected {
                        ui.colored_label(egui::Color32::GREEN, "● ");
                        ui.label(&self.db_status);
                    } else {
                        ui.colored_label(egui::Color32::RED, "● ");
                        ui.label(&self.db_status);
                    }
                });
            });
            ui.add_space(4.0);
        });

        // Bottom panel - Log
        egui::TopBottomPanel::bottom("log").resizable(true).min_height(150.0).show(ctx, |ui| {
            ui.add_space(4.0);
            ui.heading("Log");
            ui.separator();

            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
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

        // Central panel - Main content
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(8.0);

                // File Selection Panel
                ui.group(|ui| {
                    ui.heading("CSV File Selection");
                    ui.add_space(4.0);

                    egui::Grid::new("file_grid")
                        .num_columns(3)
                        .spacing([8.0, 8.0])
                        .show(ui, |ui| {
                            // Organizations
                            ui.label("Organizations:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.org_path)
                                    .desired_width(400.0)
                                    .hint_text("Select organizations.csv"),
                            );
                            if ui.button("Browse...").clicked() {
                                self.browse_file("organizations");
                            }
                            ui.end_row();

                            // Regions (Optional)
                            ui.label("Regions (Optional):");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.region_path)
                                    .desired_width(400.0)
                                    .hint_text("Select regions.csv (optional)"),
                            );
                            if ui.button("Browse...").clicked() {
                                self.browse_file("regions");
                            }
                            ui.end_row();

                            // Facilities
                            ui.label("Facilities:");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.facility_path)
                                    .desired_width(400.0)
                                    .hint_text("Select facilities.csv"),
                            );
                            if ui.button("Browse...").clicked() {
                                self.browse_file("facilities");
                            }
                            ui.end_row();

                            // Providers (Optional)
                            ui.label("Providers (Optional):");
                            ui.add(
                                egui::TextEdit::singleline(&mut self.provider_path)
                                    .desired_width(400.0)
                                    .hint_text("Select providers.csv (optional)"),
                            );
                            if ui.button("Browse...").clicked() {
                                self.browse_file("providers");
                            }
                            ui.end_row();
                        });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Load from Directory...").clicked() {
                            self.browse_directory();
                        }
                        if ui.button("Generate Templates...").clicked() {
                            self.generate_templates();
                        }
                        if ui.button("Clear All").clicked() {
                            self.clear_files();
                        }
                    });
                });

                ui.add_space(12.0);

                // Validation/Import Status Panel
                ui.group(|ui| {
                    ui.heading("Status");
                    ui.add_space(4.0);

                    match &self.app_state {
                        AppState::Idle => {
                            ui.label("Ready. Select CSV files and click Validate & Import.");
                        }
                        AppState::Validating => {
                            ui.label("Validating data...");
                            ui.add(egui::Spinner::new());
                        }
                        AppState::ValidationSuccess => {
                            ui.colored_label(egui::Color32::GREEN, "Validation successful!");
                            ui.add_space(4.0);
                            egui::Grid::new("validation_results")
                                .num_columns(2)
                                .show(ui, |ui| {
                                    ui.label("Organizations:");
                                    ui.label(self.org_count.to_string());
                                    ui.end_row();
                                    ui.label("Regions:");
                                    ui.label(self.region_count.to_string());
                                    ui.end_row();
                                    ui.label("Facilities:");
                                    ui.label(self.facility_count.to_string());
                                    ui.end_row();
                                    ui.label("Providers:");
                                    ui.label(self.provider_count.to_string());
                                    ui.end_row();
                                });
                        }
                        AppState::ValidationError(err) => {
                            ui.colored_label(egui::Color32::RED, "Validation failed!");
                            ui.add_space(4.0);
                            ui.label(err);
                        }
                        AppState::Importing => {
                            ui.label("Importing data to database...");
                            ui.add(egui::ProgressBar::new(self.import_progress).show_percentage());
                        }
                        AppState::ImportSuccess => {
                            ui.colored_label(egui::Color32::GREEN, "Import completed successfully!");
                            if let Some(ref results) = self.import_results {
                                ui.add_space(4.0);
                                egui::Grid::new("import_results")
                                    .num_columns(2)
                                    .show(ui, |ui| {
                                        ui.label("Organizations:");
                                        ui.label(results.organizations_inserted.to_string());
                                        ui.end_row();
                                        ui.label("Regions:");
                                        ui.label(results.regions_inserted.to_string());
                                        ui.end_row();
                                        ui.label("Facilities:");
                                        ui.label(results.facilities_inserted.to_string());
                                        ui.end_row();
                                        ui.label("Providers:");
                                        ui.label(results.providers_inserted.to_string());
                                        ui.end_row();
                                        ui.label("Total:");
                                        ui.strong(results.total_inserted().to_string());
                                        ui.end_row();
                                    });
                            }
                        }
                        AppState::ImportError(err) => {
                            ui.colored_label(egui::Color32::RED, "Import failed!");
                            ui.add_space(4.0);
                            ui.label(err);
                        }
                    }
                });

                ui.add_space(12.0);

                // Database Configuration Help (shown when not connected)
                if !self.db_connected {
                    ui.group(|ui| {
                        ui.heading("Database Configuration Required");
                        ui.add_space(4.0);
                        ui.label("The database has not been configured yet.");
                        ui.add_space(4.0);
                        ui.label("To configure the database connection:");
                        ui.label("1. Close this application");
                        ui.label("2. Run the Configuration Wizard from:");
                        ui.label("   Start Menu > Professional SMART > Configuration Wizard");
                        ui.label("3. Follow the wizard to set up your database connection");
                        ui.label("4. Reopen this application");
                    });
                    ui.add_space(12.0);
                }

                // Action Buttons
                ui.horizontal(|ui| {
                    // Only require organizations and facilities (regions and providers are optional)
                    let can_validate = !self.org_path.is_empty()
                        && !self.facility_path.is_empty()
                        && self.db_connected
                        && matches!(
                            self.app_state,
                            AppState::Idle | AppState::ValidationError(_) | AppState::ImportSuccess | AppState::ImportError(_)
                        );

                    if ui
                        .add_enabled(can_validate, egui::Button::new("Validate & Import"))
                        .clicked()
                    {
                        self.validate_data();
                    }

                    if matches!(self.app_state, AppState::ImportSuccess) {
                        if ui.button("Import More Data").clicked() {
                            self.clear_files();
                            self.app_state = AppState::Idle;
                            self.org_count = 0;
                            self.region_count = 0;
                            self.facility_count = 0;
                            self.provider_count = 0;
                            self.import_results = None;
                            self.import_progress = 0.0;
                        }
                    }
                });

                ui.add_space(8.0);
            });
        });
    }
}
