#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

extern crate native_windows_gui as nwg;
extern crate native_windows_derive as nwd;

use nwd::NwgUi;
use nwg::NativeUi;
use std::cell::RefCell;
use std::sync::mpsc;
use pro_data_loader::{Config, Facility, ImportResults, Organization, Provider, Region};

// Base dimensions at 96 DPI (100% scaling)
// These are scaled at runtime based on actual DPI
const BASE_WINDOW_WIDTH: i32 = 900;
const BASE_WINDOW_HEIGHT: i32 = 680;
const BASE_MARGIN: i32 = 16;
const BASE_LABEL_WIDTH: i32 = 150;
const BASE_TEXT_WIDTH: i32 = 520;
const BASE_BROWSE_BTN_WIDTH: i32 = 80;
const BASE_ACTION_BTN_WIDTH: i32 = 150;
const BASE_ROW_HEIGHT: i32 = 34;
const BASE_CONTROL_HEIGHT: i32 = 26;
const BASE_BTN_HEIGHT: i32 = 30;
const BASE_SECTION_GAP: i32 = 20;

// Color constants for status indicators (RGB)
const COLOR_SUCCESS: [u8; 3] = [34, 139, 34];    // Forest Green
const COLOR_WARNING: [u8; 3] = [184, 134, 11];   // Dark Goldenrod
const COLOR_ERROR: [u8; 3] = [178, 34, 34];      // Firebrick Red
const COLOR_INFO: [u8; 3] = [70, 130, 180];      // Steel Blue
const COLOR_HEADER: [u8; 3] = [25, 25, 112];     // Midnight Blue

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

#[derive(Default, NwgUi)]
pub struct DataLoaderApp {
    // Configuration
    config: RefCell<Option<Config>>,
    app_state: RefCell<AppState>,
    validation_data: RefCell<Option<ValidationData>>,
    import_results: RefCell<Option<ImportResults>>,

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
    #[nwg_control(size: (900, 680), position: (100, 100), title: "Professional SMART - Master Data Loader", flags: "WINDOW|VISIBLE|MINIMIZE_BOX|MAXIMIZE_BOX")]
    #[nwg_events(OnWindowClose: [DataLoaderApp::exit], OnInit: [DataLoaderApp::on_init])]
    window: nwg::Window,

    // Database connection status - RichLabel for colored text
    #[nwg_control(parent: window, position: (16, 16), size: (860, 22), flags: "VISIBLE")]
    rich_db_status: nwg::RichLabel,

    // Section header
    #[nwg_control(parent: window, text: "CSV File Selection", position: (16, 48), size: (200, 22))]
    lbl_files_header: nwg::Label,

    // File input rows - label width 130px accommodates "Providers (Optional):"
    // Organizations row
    #[nwg_control(parent: window, text: "Organizations:", position: (12, 68), size: (130, 20))]
    lbl_org: nwg::Label,
    #[nwg_control(parent: window, text: "", position: (148, 66), size: (540, 24))]
    txt_org: nwg::TextInput,
    #[nwg_control(parent: window, text: "Browse...", position: (696, 64), size: (85, 28))]
    #[nwg_events(OnButtonClick: [DataLoaderApp::browse_org])]
    btn_org: nwg::Button,

    // Regions row
    #[nwg_control(parent: window, text: "Regions (Optional):", position: (12, 98), size: (130, 20))]
    lbl_region: nwg::Label,
    #[nwg_control(parent: window, text: "", position: (148, 96), size: (540, 24))]
    txt_region: nwg::TextInput,
    #[nwg_control(parent: window, text: "Browse...", position: (696, 94), size: (85, 28))]
    #[nwg_events(OnButtonClick: [DataLoaderApp::browse_region])]
    btn_region: nwg::Button,

    // Facilities row
    #[nwg_control(parent: window, text: "Facilities:", position: (12, 128), size: (130, 20))]
    lbl_facility: nwg::Label,
    #[nwg_control(parent: window, text: "", position: (148, 126), size: (540, 24))]
    txt_facility: nwg::TextInput,
    #[nwg_control(parent: window, text: "Browse...", position: (696, 124), size: (85, 28))]
    #[nwg_events(OnButtonClick: [DataLoaderApp::browse_facility])]
    btn_facility: nwg::Button,

    // Providers row
    #[nwg_control(parent: window, text: "Providers (Optional):", position: (12, 158), size: (130, 20))]
    lbl_provider: nwg::Label,
    #[nwg_control(parent: window, text: "", position: (148, 156), size: (540, 24))]
    txt_provider: nwg::TextInput,
    #[nwg_control(parent: window, text: "Browse...", position: (696, 154), size: (85, 28))]
    #[nwg_events(OnButtonClick: [DataLoaderApp::browse_provider])]
    btn_provider: nwg::Button,

    // Action buttons row - button widths accommodate text
    #[nwg_control(parent: window, text: "Load from Directory...", position: (12, 196), size: (150, 28))]
    #[nwg_events(OnButtonClick: [DataLoaderApp::browse_directory])]
    btn_load_dir: nwg::Button,

    #[nwg_control(parent: window, text: "Generate Templates...", position: (170, 196), size: (150, 28))]
    #[nwg_events(OnButtonClick: [DataLoaderApp::generate_templates])]
    btn_templates: nwg::Button,

    #[nwg_control(parent: window, text: "Clear All", position: (328, 196), size: (85, 28))]
    #[nwg_events(OnButtonClick: [DataLoaderApp::clear_files])]
    btn_clear: nwg::Button,

    // Status section
    #[nwg_control(parent: window, text: "Status", position: (16, 260), size: (60, 22))]
    lbl_status_header: nwg::Label,

    #[nwg_control(parent: window, position: (16, 286), size: (860, 24), flags: "VISIBLE")]
    rich_status: nwg::RichLabel,

    #[nwg_control(parent: window, position: (16, 316), size: (860, 22), range: 0..100)]
    progress_bar: nwg::ProgressBar,

    // Results display area - RichLabel for formatted results
    #[nwg_control(parent: window, position: (16, 344), size: (860, 90), flags: "VISIBLE")]
    rich_results: nwg::RichLabel,

    // Import action buttons
    #[nwg_control(parent: window, text: "Validate && Import", position: (12, 402), size: (150, 30))]
    #[nwg_events(OnButtonClick: [DataLoaderApp::validate_and_import])]
    btn_validate: nwg::Button,

    #[nwg_control(parent: window, text: "Import More Data", position: (170, 402), size: (140, 30))]
    #[nwg_events(OnButtonClick: [DataLoaderApp::import_more])]
    btn_import_more: nwg::Button,

    // Log section - RichTextBox for colored log entries
    #[nwg_control(parent: window, text: "Activity Log", position: (16, 480), size: (100, 22))]
    lbl_log_header: nwg::Label,

    #[nwg_control(parent: window, position: (16, 506), size: (860, 150), flags: "VISIBLE|VSCROLL", readonly: true)]
    log_box: nwg::RichTextBox,

    // Timer for processing background messages
    #[nwg_control(interval: std::time::Duration::from_millis(100))]
    #[nwg_events(OnTimerTick: [DataLoaderApp::process_messages])]
    timer: nwg::AnimationTimer,

    // File dialogs
    #[nwg_resource(title: "Select CSV File", filters: "CSV Files(*.csv)|All Files(*.*)")]
    file_dialog: nwg::FileDialog,

    #[nwg_resource(title: "Select Directory", action: nwg::FileDialogAction::OpenDirectory)]
    folder_dialog: nwg::FileDialog,
}

impl DataLoaderApp {
    fn on_init(&self) {
        // Apply DPI scaling to all controls
        self.apply_dpi_scaling();

        // Apply fonts to controls
        self.lbl_files_header.set_font(Some(&self.header_font));
        self.lbl_status_header.set_font(Some(&self.header_font));
        self.lbl_log_header.set_font(Some(&self.header_font));
        self.log_box.set_font(Some(&self.log_font));

        // Initialize status with styled text
        self.set_status("Ready. Select CSV files and click Validate & Import.", LogLevel::Info);

        self.timer.start();
        self.load_config();
        self.add_log(LogLevel::Info, "Application started");
    }

    /// Set status text with appropriate color styling
    fn set_status(&self, text: &str, level: LogLevel) {
        let color = match level {
            LogLevel::Success => COLOR_SUCCESS,
            LogLevel::Warning => COLOR_WARNING,
            LogLevel::Error => COLOR_ERROR,
            LogLevel::Info => COLOR_INFO,
        };
        self.rich_status.set_text(text);
        let text_len = text.len() as u32;
        self.rich_status.set_char_format(0..text_len, &nwg::CharFormat {
            effects: None,
            height: None,
            y_offset: None,
            text_color: Some(color),
            font_face_name: Some("Segoe UI".to_string()),
            underline_type: None,
        });
    }

    /// Set database status with colored text
    fn set_db_status(&self, text: &str, connected: bool) {
        let color = if connected { COLOR_SUCCESS } else { COLOR_WARNING };
        self.rich_db_status.set_text(text);
        let text_len = text.len() as u32;
        self.rich_db_status.set_char_format(0..text_len, &nwg::CharFormat {
            effects: Some(nwg::CharEffects::BOLD),
            height: None,
            y_offset: None,
            text_color: Some(color),
            font_face_name: Some("Segoe UI".to_string()),
            underline_type: None,
        });
    }

    /// Set results text with formatting
    fn set_results(&self, text: &str) {
        self.rich_results.set_text(text);
        let text_len = text.len() as u32;
        self.rich_results.set_char_format(0..text_len, &nwg::CharFormat {
            effects: None,
            height: None,
            y_offset: None,
            text_color: Some(COLOR_SUCCESS),
            font_face_name: Some("Segoe UI".to_string()),
            underline_type: None,
        });
    }

    fn apply_dpi_scaling(&self) {
        // Get the current scale factor (1.0 at 96 DPI, 1.25 at 120 DPI, etc.)
        let scale = nwg::scale_factor() as f32;

        // Helper to scale a value
        let s = |v: i32| -> i32 { (v as f32 * scale) as i32 };

        // Calculate scaled dimensions
        let margin = s(BASE_MARGIN);
        let label_width = s(BASE_LABEL_WIDTH);
        let text_width = s(BASE_TEXT_WIDTH);
        let browse_btn_width = s(BASE_BROWSE_BTN_WIDTH);
        let action_btn_width = s(BASE_ACTION_BTN_WIDTH);
        let row_height = s(BASE_ROW_HEIGHT);
        let control_height = s(BASE_CONTROL_HEIGHT);
        let btn_height = s(BASE_BTN_HEIGHT);
        let section_gap = s(BASE_SECTION_GAP);
        let window_width = s(BASE_WINDOW_WIDTH);
        let window_height = s(BASE_WINDOW_HEIGHT);

        // Text input starts after label
        let text_x = margin + label_width + margin / 2;
        // Browse button at end
        let browse_x = text_x + text_width + margin / 2;
        // Full content width
        let content_width = window_width - margin * 2;

        // Resize window
        self.window.set_size(window_width as u32, window_height as u32);

        // Row positions (Y coordinates)
        let mut y = margin;

        // Database status - full width with RichLabel
        self.rich_db_status.set_position(margin, y);
        self.rich_db_status.set_size(content_width as u32, s(24) as u32);
        y += s(32);

        // Section header
        self.lbl_files_header.set_position(margin, y);
        self.lbl_files_header.set_size(s(200) as u32, s(24) as u32);
        y += s(30);

        // Organizations row
        self.lbl_org.set_position(margin, y + 3);
        self.lbl_org.set_size(label_width as u32, s(22) as u32);
        self.txt_org.set_position(text_x, y);
        self.txt_org.set_size(text_width as u32, control_height as u32);
        self.btn_org.set_position(browse_x, y - 2);
        self.btn_org.set_size(browse_btn_width as u32, btn_height as u32);
        y += row_height;

        // Regions row
        self.lbl_region.set_position(margin, y + 3);
        self.lbl_region.set_size(label_width as u32, s(22) as u32);
        self.txt_region.set_position(text_x, y);
        self.txt_region.set_size(text_width as u32, control_height as u32);
        self.btn_region.set_position(browse_x, y - 2);
        self.btn_region.set_size(browse_btn_width as u32, btn_height as u32);
        y += row_height;

        // Facilities row
        self.lbl_facility.set_position(margin, y + 3);
        self.lbl_facility.set_size(label_width as u32, s(22) as u32);
        self.txt_facility.set_position(text_x, y);
        self.txt_facility.set_size(text_width as u32, control_height as u32);
        self.btn_facility.set_position(browse_x, y - 2);
        self.btn_facility.set_size(browse_btn_width as u32, btn_height as u32);
        y += row_height;

        // Providers row
        self.lbl_provider.set_position(margin, y + 3);
        self.lbl_provider.set_size(label_width as u32, s(22) as u32);
        self.txt_provider.set_position(text_x, y);
        self.txt_provider.set_size(text_width as u32, control_height as u32);
        self.btn_provider.set_position(browse_x, y - 2);
        self.btn_provider.set_size(browse_btn_width as u32, btn_height as u32);
        y += row_height + section_gap;

        // Action buttons row
        let mut btn_x = margin;
        self.btn_load_dir.set_position(btn_x, y);
        self.btn_load_dir.set_size(action_btn_width as u32, btn_height as u32);
        btn_x += action_btn_width + margin;

        self.btn_templates.set_position(btn_x, y);
        self.btn_templates.set_size(action_btn_width as u32, btn_height as u32);
        btn_x += action_btn_width + margin;

        self.btn_clear.set_position(btn_x, y);
        self.btn_clear.set_size(s(90) as u32, btn_height as u32);
        y += btn_height + section_gap;

        // Status section
        self.lbl_status_header.set_position(margin, y);
        self.lbl_status_header.set_size(s(80) as u32, s(24) as u32);
        y += s(28);

        self.rich_status.set_position(margin, y);
        self.rich_status.set_size(content_width as u32, s(24) as u32);
        y += s(30);

        self.progress_bar.set_position(margin, y);
        self.progress_bar.set_size(content_width as u32, s(22) as u32);
        y += s(30);

        // Results display area
        self.rich_results.set_position(margin, y);
        self.rich_results.set_size(content_width as u32, s(90) as u32);
        y += s(98);

        // Import action buttons
        self.btn_validate.set_position(margin, y);
        self.btn_validate.set_size(action_btn_width as u32, s(32) as u32);

        self.btn_import_more.set_position(margin + action_btn_width + margin, y);
        self.btn_import_more.set_size(s(140) as u32, s(32) as u32);
        y += s(44);

        // Log section
        self.lbl_log_header.set_position(margin, y);
        self.lbl_log_header.set_size(s(100) as u32, s(24) as u32);
        y += s(28);

        // Log box fills remaining space
        let log_height = window_height - y - margin;
        self.log_box.set_position(margin, y);
        self.log_box.set_size(content_width as u32, log_height as u32);
    }

    fn load_config(&self) {
        match Config::load() {
            Ok(config) => {
                self.set_db_status(&format!("Database: {}", config.masked_url()), true);
                self.add_log(LogLevel::Success, "Database configuration loaded successfully");
                *self.config.borrow_mut() = Some(config);
            }
            Err(e) => {
                let error_msg = e.to_string();
                if error_msg.contains("Could not find .env file") {
                    self.set_db_status("Database: Not configured", false);
                    self.add_log(LogLevel::Error, "Database configuration file not found");
                    self.add_log(LogLevel::Info, "Please run the Configuration Wizard");
                } else {
                    self.set_db_status("Database: Configuration error", false);
                    self.add_log(LogLevel::Error, &format!("Config error: {}", error_msg));
                }
            }
        }
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

    fn browse_org(&self) {
        if self.file_dialog.run(Some(&self.window)) {
            if let Ok(path) = self.file_dialog.get_selected_item() {
                self.txt_org.set_text(&path.to_string_lossy());
                self.add_log(LogLevel::Info, &format!("Selected organizations: {}", path.display()));
            }
        }
    }

    fn browse_region(&self) {
        if self.file_dialog.run(Some(&self.window)) {
            if let Ok(path) = self.file_dialog.get_selected_item() {
                self.txt_region.set_text(&path.to_string_lossy());
                self.add_log(LogLevel::Info, &format!("Selected regions: {}", path.display()));
            }
        }
    }

    fn browse_facility(&self) {
        if self.file_dialog.run(Some(&self.window)) {
            if let Ok(path) = self.file_dialog.get_selected_item() {
                self.txt_facility.set_text(&path.to_string_lossy());
                self.add_log(LogLevel::Info, &format!("Selected facilities: {}", path.display()));
            }
        }
    }

    fn browse_provider(&self) {
        if self.file_dialog.run(Some(&self.window)) {
            if let Ok(path) = self.file_dialog.get_selected_item() {
                self.txt_provider.set_text(&path.to_string_lossy());
                self.add_log(LogLevel::Info, &format!("Selected providers: {}", path.display()));
            }
        }
    }

    fn browse_directory(&self) {
        if self.folder_dialog.run(Some(&self.window)) {
            if let Ok(os_path) = self.folder_dialog.get_selected_item() {
                let path = std::path::PathBuf::from(&os_path);
                let org = path.join("organizations.csv");
                let region = path.join("regions.csv");
                let facility = path.join("facilities.csv");
                let provider = path.join("providers.csv");

                let mut loaded = 0;
                if org.exists() {
                    self.txt_org.set_text(&org.to_string_lossy());
                    loaded += 1;
                }
                if region.exists() {
                    self.txt_region.set_text(&region.to_string_lossy());
                    loaded += 1;
                }
                if facility.exists() {
                    self.txt_facility.set_text(&facility.to_string_lossy());
                    loaded += 1;
                }
                if provider.exists() {
                    self.txt_provider.set_text(&provider.to_string_lossy());
                    loaded += 1;
                }

                if loaded > 0 {
                    self.add_log(LogLevel::Info, &format!("Loaded {} CSV file(s) from directory", loaded));
                } else {
                    self.add_log(LogLevel::Warning, "No CSV files found in directory");
                }
            }
        }
    }

    fn generate_templates(&self) {
        if self.folder_dialog.run(Some(&self.window)) {
            if let Ok(os_path) = self.folder_dialog.get_selected_item() {
                let path = std::path::PathBuf::from(&os_path);
                match pro_data_loader::templates::generate_templates(&path) {
                    Ok(_) => {
                        self.add_log(LogLevel::Success, &format!("Templates generated in: {}", path.display()));
                        nwg::simple_message("Success", &format!("Templates generated in:\n{}", path.display()));
                    }
                    Err(e) => {
                        self.add_log(LogLevel::Error, &format!("Failed to generate templates: {}", e));
                    }
                }
            }
        }
    }

    fn clear_files(&self) {
        self.txt_org.set_text("");
        self.txt_region.set_text("");
        self.txt_facility.set_text("");
        self.txt_provider.set_text("");
        self.add_log(LogLevel::Info, "Cleared all file selections");
    }

    fn validate_and_import(&self) {
        let org_path = self.txt_org.text();
        let facility_path = self.txt_facility.text();

        if org_path.is_empty() || facility_path.is_empty() {
            self.add_log(LogLevel::Error, "Organizations and Facilities CSV files are required");
            nwg::simple_message("Missing Files", "Organizations and Facilities CSV files are required.\nRegions and Providers are optional.");
            return;
        }

        if self.config.borrow().is_none() {
            self.add_log(LogLevel::Error, "Database not configured");
            nwg::simple_message("Configuration Required", "Please configure the database connection first.");
            return;
        }

        *self.app_state.borrow_mut() = AppState::Validating;
        self.set_status("Validating data...", LogLevel::Info);
        self.progress_bar.set_pos(10);
        self.add_log(LogLevel::Info, "Starting validation...");

        let (tx, rx) = mpsc::channel();
        *self.task_receiver.borrow_mut() = Some(rx);

        let region_path = self.txt_region.text();
        let provider_path = self.txt_provider.text();
        let database_url = self.config.borrow().as_ref().unwrap().database_url.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                tx.send(TaskMessage::Log(LogLevel::Info, "Parsing CSV files...".to_string())).ok();

                // Parse files
                let organizations = match pro_data_loader::csv_parser::parse_organizations(&org_path) {
                    Ok(o) => o,
                    Err(e) => {
                        tx.send(TaskMessage::ValidationComplete(Err(format!("Organizations: {}", e)))).ok();
                        return;
                    }
                };
                tx.send(TaskMessage::Log(LogLevel::Info, format!("Parsed {} organizations", organizations.len()))).ok();

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

                let facilities = match pro_data_loader::csv_parser::parse_facilities(&facility_path) {
                    Ok(f) => f,
                    Err(e) => {
                        tx.send(TaskMessage::ValidationComplete(Err(format!("Facilities: {}", e)))).ok();
                        return;
                    }
                };
                tx.send(TaskMessage::Log(LogLevel::Info, format!("Parsed {} facilities", facilities.len()))).ok();

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

    fn import_more(&self) {
        self.clear_files();
        *self.app_state.borrow_mut() = AppState::Idle;
        *self.validation_data.borrow_mut() = None;
        *self.import_results.borrow_mut() = None;
        self.set_status("Ready. Select CSV files and click Validate & Import.", LogLevel::Info);
        self.rich_results.set_text("");
        self.progress_bar.set_pos(0);
    }

    fn process_messages(&self) {
        let receiver = self.task_receiver.borrow();
        if let Some(ref rx) = *receiver {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    TaskMessage::Log(level, message) => {
                        self.add_log(level, &message);
                    }
                    TaskMessage::ValidationComplete(result) => {
                        match result {
                            Ok(data) => {
                                let total = data.organizations.len() + data.regions.len() + data.facilities.len() + data.providers.len();
                                *self.app_state.borrow_mut() = AppState::ValidationSuccess;
                                self.set_status("Validation successful! Importing...", LogLevel::Success);
                                self.progress_bar.set_pos(30);
                                self.add_log(LogLevel::Success, &format!("Validation successful! {} records to import", total));
                                *self.validation_data.borrow_mut() = Some(data);
                            }
                            Err(e) => {
                                *self.app_state.borrow_mut() = AppState::ValidationError(e.clone());
                                self.set_status(&format!("Validation failed: {}", e), LogLevel::Error);
                                self.progress_bar.set_pos(0);
                                self.add_log(LogLevel::Error, &format!("Validation failed: {}", e));
                            }
                        }
                    }
                    TaskMessage::ImportProgress(progress) => {
                        self.progress_bar.set_pos((progress * 100.0) as u32);
                    }
                    TaskMessage::ImportComplete(result) => {
                        match result {
                            Ok(results) => {
                                *self.app_state.borrow_mut() = AppState::ImportSuccess;
                                self.set_status("Import completed successfully!", LogLevel::Success);
                                self.progress_bar.set_pos(100);
                                self.set_results(&format!(
                                    "Organizations: {}\r\nRegions: {}\r\nFacilities: {}\r\nProviders: {}\r\nTotal: {}",
                                    results.organizations_inserted,
                                    results.regions_inserted,
                                    results.facilities_inserted,
                                    results.providers_inserted,
                                    results.total_inserted()
                                ));
                                self.add_log(LogLevel::Success, &format!("Import completed! {} total records", results.total_inserted()));
                                *self.import_results.borrow_mut() = Some(results);
                                nwg::simple_message("Import Complete", "Data import completed successfully!");
                            }
                            Err(e) => {
                                *self.app_state.borrow_mut() = AppState::ImportError(e.clone());
                                self.set_status(&format!("Import failed: {}", e), LogLevel::Error);
                                self.progress_bar.set_pos(0);
                                self.add_log(LogLevel::Error, &format!("Import failed: {}", e));
                            }
                        }
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

fn main() {
    // Initialize Native Windows GUI
    nwg::init().expect("Failed to initialize NWG");
    nwg::Font::set_global_family("Segoe UI").expect("Failed to set font");

    // Build and run the application
    let _app = DataLoaderApp::build_ui(Default::default()).expect("Failed to build UI");
    nwg::dispatch_thread_events();
}
