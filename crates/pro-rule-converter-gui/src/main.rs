#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

extern crate native_windows_gui as nwg;
extern crate native_windows_derive as nwd;

use nwd::NwgUi;
use nwg::NativeUi;
use std::cell::RefCell;
use std::fs;
use std::sync::mpsc;
use anyhow::{anyhow, Result};
use serde::Deserialize;

mod converter;
mod mssql;

use converter::generate_sql_for_rule;
use mssql::MsSqlClient;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub query: QueryConfig,
    pub output: OutputConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub server: String,
    pub port: u16,
    pub database: String,
    pub auth_type: String,
    pub username: Option<String>,
    pub password: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QueryConfig {
    pub sql: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OutputConfig {
    pub output_directory: String,
    pub flag_category: String,
}

#[derive(Debug, Clone)]
pub struct RuleRow {
    pub filter_number: String,
    pub filter_name: String,
    pub description: String,
    pub definition: String,
    pub selected: bool,
}

#[derive(Debug)]
enum TaskMessage {
    Log(String),
    Error(String),
    RulesLoaded(Vec<RuleRow>),
}

fn load_config() -> Result<Config> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().ok_or_else(|| anyhow!("No parent dir"))?;
    let config_path = exe_dir.join("rule-converter-config.toml");

    if !config_path.exists() {
        let current_dir = std::env::current_dir()?;
        let config_path = current_dir.join("rule-converter-config.toml");
        if config_path.exists() {
            let content = fs::read_to_string(&config_path)?;
            return Ok(toml::from_str(&content)?);
        }
        return Err(anyhow!("Config file not found: rule-converter-config.toml"));
    }

    let content = fs::read_to_string(&config_path)?;
    Ok(toml::from_str(&content)?)
}

/// Truncate string safely at char boundary
fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}

#[derive(Default, NwgUi)]
pub struct RuleConverterApp {
    config: RefCell<Option<Config>>,
    rules: RefCell<Vec<RuleRow>>,
    task_receiver: RefCell<Option<mpsc::Receiver<TaskMessage>>>,
    /// Flag to prevent selection event handling during bulk operations
    bulk_operation: RefCell<bool>,

    #[nwg_resource(family: "Segoe UI Semibold", size: 16)]
    header_font: nwg::Font,

    #[nwg_resource(family: "Segoe UI", size: 14)]
    body_font: nwg::Font,

    #[nwg_resource(family: "Consolas", size: 12)]
    log_font: nwg::Font,

    #[nwg_control(size: (900, 650), position: (100, 100), title: "Rule Converter - MS SQL to COMPOSITE Template", flags: "WINDOW|VISIBLE|MINIMIZE_BOX")]
    #[nwg_events(OnWindowClose: [RuleConverterApp::exit], OnInit: [RuleConverterApp::on_init])]
    window: nwg::Window,

    #[nwg_control(parent: window, text: "Database Connection", position: (16, 12), size: (200, 22))]
    lbl_connection_header: nwg::Label,

    #[nwg_control(parent: window, text: "Server:", position: (16, 40), size: (50, 20))]
    lbl_server: nwg::Label,

    #[nwg_control(parent: window, text: "localhost", position: (70, 38), size: (150, 24))]
    txt_server: nwg::TextInput,

    #[nwg_control(parent: window, text: "Port:", position: (230, 40), size: (35, 20))]
    lbl_port: nwg::Label,

    #[nwg_control(parent: window, text: "1433", position: (270, 38), size: (55, 24))]
    txt_port: nwg::TextInput,

    #[nwg_control(parent: window, text: "Database:", position: (340, 40), size: (65, 20))]
    lbl_database: nwg::Label,

    #[nwg_control(parent: window, text: "", position: (410, 38), size: (140, 24))]
    txt_database: nwg::TextInput,

    #[nwg_control(parent: window, text: "Username:", position: (565, 40), size: (70, 20))]
    lbl_username: nwg::Label,

    #[nwg_control(parent: window, text: "", position: (640, 38), size: (110, 24))]
    txt_username: nwg::TextInput,

    #[nwg_control(parent: window, text: "Password:", position: (760, 40), size: (65, 20))]
    lbl_password: nwg::Label,

    #[nwg_control(parent: window, text: "", position: (830, 38), size: (50, 24), flags: "VISIBLE|TAB_STOP", password: Some('*'))]
    txt_password: nwg::TextInput,

    #[nwg_control(parent: window, text: "Connect & Load Rules", position: (16, 70), size: (150, 28))]
    #[nwg_events(OnButtonClick: [RuleConverterApp::connect_and_load])]
    btn_connect: nwg::Button,

    #[nwg_control(parent: window, text: "Rules (select rows to export)", position: (16, 106), size: (240, 22))]
    lbl_rules_header: nwg::Label,

    #[nwg_control(parent: window, position: (16, 130), size: (860, 270), flags: "VISIBLE|TAB_STOP", list_style: ListViewStyle::Detailed, ex_flags: ListViewExFlags::FULL_ROW_SELECT)]
    list_rules: nwg::ListView,

    #[nwg_control(parent: window, text: "Select All", position: (16, 410), size: (100, 28))]
    #[nwg_events(OnButtonClick: [RuleConverterApp::select_all])]
    btn_select_all: nwg::Button,

    #[nwg_control(parent: window, text: "Deselect All", position: (126, 410), size: (100, 28))]
    #[nwg_events(OnButtonClick: [RuleConverterApp::deselect_all])]
    btn_deselect_all: nwg::Button,

    #[nwg_control(parent: window, text: "Export Selected to SQL", position: (720, 410), size: (156, 28))]
    #[nwg_events(OnButtonClick: [RuleConverterApp::export_selected])]
    btn_export: nwg::Button,

    #[nwg_control(parent: window, text: "Log", position: (16, 450), size: (100, 22))]
    lbl_log_header: nwg::Label,

    #[nwg_control(parent: window, text: "", position: (16, 474), size: (860, 160), flags: "VISIBLE|VSCROLL|AUTOVSCROLL", readonly: true)]
    txt_log: nwg::TextBox,

    #[nwg_control(interval: std::time::Duration::from_millis(100))]
    #[nwg_events(OnTimerTick: [RuleConverterApp::check_messages])]
    timer: nwg::AnimationTimer,
}

impl RuleConverterApp {
    fn on_init(&self) {
        self.lbl_connection_header.set_font(Some(&self.header_font));
        self.lbl_rules_header.set_font(Some(&self.header_font));
        self.lbl_log_header.set_font(Some(&self.header_font));
        self.txt_log.set_font(Some(&self.log_font));
        self.lbl_server.set_font(Some(&self.body_font));
        self.lbl_database.set_font(Some(&self.body_font));

        self.list_rules.insert_column(nwg::InsertListViewColumn {
            index: Some(0),
            fmt: None,
            width: Some(100),
            text: Some("Rule Code".into()),
        });
        self.list_rules.insert_column(nwg::InsertListViewColumn {
            index: Some(1),
            fmt: None,
            width: Some(200),
            text: Some("Rule Name".into()),
        });
        self.list_rules.insert_column(nwg::InsertListViewColumn {
            index: Some(2),
            fmt: None,
            width: Some(200),
            text: Some("Description".into()),
        });
        self.list_rules.insert_column(nwg::InsertListViewColumn {
            index: Some(3),
            fmt: None,
            width: Some(340),
            text: Some("Definition".into()),
        });

        match load_config() {
            Ok(config) => {
                self.txt_server.set_text(&config.database.server);
                self.txt_port.set_text(&config.database.port.to_string());
                self.txt_database.set_text(&config.database.database);
                if let Some(ref username) = config.database.username {
                    self.txt_username.set_text(username);
                }
                if let Some(ref password) = config.database.password {
                    self.txt_password.set_text(password);
                }
                *self.config.borrow_mut() = Some(config);
                self.log("Config loaded successfully");
            }
            Err(e) => {
                self.log(&format!("Warning: Could not load config: {}", e));
                self.log("Using default settings. Create rule-converter-config.toml to customize.");
            }
        }

        self.timer.start();
    }

    fn log(&self, msg: &str) {
        let current = self.txt_log.text();
        let new_text = if current.is_empty() {
            msg.to_string()
        } else {
            format!("{}\r\n{}", current, msg)
        };
        self.txt_log.set_text(&new_text);
        let len = new_text.len() as u32;
        self.txt_log.set_selection(len..len);
    }

    fn connect_and_load(&self) {
        let server = self.txt_server.text();
        let database = self.txt_database.text();
        let username = self.txt_username.text();
        let password = self.txt_password.text();

        if server.is_empty() || database.is_empty() {
            self.log("ERROR: Server and Database are required");
            return;
        }

        if username.is_empty() || password.is_empty() {
            self.log("ERROR: Username and Password are required for SQL Server Authentication");
            return;
        }

        self.log(&format!("Connecting to {}/{}...", server, database));

        let config = self.config.borrow().clone();
        let sql_query = config
            .as_ref()
            .map(|c| c.query.sql.clone())
            .unwrap_or_else(|| "SELECT 'No query configured' as FilterNumber, '' as FilterName, '' as FilterDescription, '' as definition".to_string());

        let (tx, rx) = mpsc::channel();
        *self.task_receiver.borrow_mut() = Some(rx);

        let server_clone = server.clone();
        let database_clone = database.clone();
        let username_clone = username.clone();
        let password_clone = password.clone();

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                match MsSqlClient::connect_with_auth(
                    &server_clone,
                    &database_clone,
                    Some(&username_clone),
                    Some(&password_clone),
                ).await {
                    Ok(mut client) => {
                        let _ = tx.send(TaskMessage::Log(format!("Connected to {}", database_clone)));

                        match client.query_rules(&sql_query).await {
                            Ok(rules) => {
                                let _ = tx.send(TaskMessage::Log(format!("Loaded {} rules", rules.len())));
                                let _ = tx.send(TaskMessage::RulesLoaded(rules));
                            }
                            Err(e) => {
                                let _ = tx.send(TaskMessage::Error(format!("Query failed: {}", e)));
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(TaskMessage::Error(format!("Connection failed: {}", e)));
                    }
                }
            });
        });
    }

    fn check_messages(&self) {
        if let Some(ref rx) = *self.task_receiver.borrow() {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    TaskMessage::Log(text) => self.log(&text),
                    TaskMessage::Error(text) => self.log(&format!("ERROR: {}", text)),
                    TaskMessage::RulesLoaded(rules) => {
                        self.populate_list_view(&rules);
                        *self.rules.borrow_mut() = rules;
                    }
                }
            }
        }
    }

    /// Populate ListView with rules - optimized for large datasets
    fn populate_list_view(&self, rules: &[RuleRow]) {
        // Set bulk operation flag to prevent event handling
        *self.bulk_operation.borrow_mut() = true;

        // Disable redraw for faster bulk insertion
        self.list_rules.set_redraw(false);
        self.list_rules.clear();

        for (i, rule) in rules.iter().enumerate() {
            // Insert row with first column
            self.list_rules.insert_item(nwg::InsertListViewItem {
                index: Some(i as i32),
                column_index: 0,
                text: Some(rule.filter_number.clone()),
                image: None,
            });

            // Set remaining columns using set_item
            self.list_rules.update_item(i, nwg::InsertListViewItem {
                index: Some(i as i32),
                column_index: 1,
                text: Some(truncate_str(&rule.filter_name, 50)),
                image: None,
            });

            self.list_rules.update_item(i, nwg::InsertListViewItem {
                index: Some(i as i32),
                column_index: 2,
                text: Some(truncate_str(&rule.description, 60)),
                image: None,
            });

            self.list_rules.update_item(i, nwg::InsertListViewItem {
                index: Some(i as i32),
                column_index: 3,
                text: Some(truncate_str(&rule.definition, 80)),
                image: None,
            });
        }

        // Re-enable redraw and refresh
        self.list_rules.set_redraw(true);
        *self.bulk_operation.borrow_mut() = false;

        self.log(&format!("Displayed {} rules in list", rules.len()));
    }

    fn select_all(&self) {
        let count = self.list_rules.len();
        if count == 0 {
            self.log("No rules to select");
            return;
        }

        self.log(&format!("Selecting all {} rules...", count));

        // Set bulk operation flag
        *self.bulk_operation.borrow_mut() = true;

        // Disable redraw during bulk selection
        self.list_rules.set_redraw(false);

        // Select all items
        for i in 0..count {
            self.list_rules.select_item(i, true);
        }

        // Re-enable redraw
        self.list_rules.set_redraw(true);
        *self.bulk_operation.borrow_mut() = false;

        self.log(&format!("Selected {} rules", count));
    }

    fn deselect_all(&self) {
        let count = self.list_rules.len();
        if count == 0 {
            return;
        }

        self.log("Deselecting all rules...");

        // Set bulk operation flag
        *self.bulk_operation.borrow_mut() = true;

        // Disable redraw during bulk deselection
        self.list_rules.set_redraw(false);

        for i in 0..count {
            self.list_rules.select_item(i, false);
        }

        // Re-enable redraw
        self.list_rules.set_redraw(true);
        *self.bulk_operation.borrow_mut() = false;

        self.log("Deselected all rules");
    }

    fn export_selected(&self) {
        // Get selected indices directly from ListView (no separate tracking needed)
        let selected_indices: Vec<usize> = self.list_rules.selected_items();

        if selected_indices.is_empty() {
            self.log("No rules selected for export");
            nwg::modal_info_message(&self.window, "No Selection", "Please select at least one rule to export.");
            return;
        }

        self.log(&format!("Exporting {} selected rules...", selected_indices.len()));

        // Collect selected rules
        let rules = self.rules.borrow();
        let selected_rules: Vec<RuleRow> = selected_indices
            .iter()
            .filter_map(|&i| rules.get(i).cloned())
            .collect();
        drop(rules);

        if selected_rules.is_empty() {
            self.log("ERROR: Could not retrieve selected rules");
            return;
        }

        // Get category from config
        let category = self.config.borrow()
            .as_ref()
            .map(|c| c.output.flag_category.clone())
            .unwrap_or_else(|| "QM".to_string());

        // File save dialog
        let mut file_dialog = nwg::FileDialog::default();
        let build_result = nwg::FileDialog::builder()
            .title("Save SQL File")
            .action(nwg::FileDialogAction::Save)
            .build(&mut file_dialog);

        if let Err(e) = build_result {
            self.log(&format!("ERROR: Failed to create file dialog: {}", e));
            return;
        }

        if file_dialog.run(Some(&self.window)) {
            match file_dialog.get_selected_item() {
                Ok(path) => {
                    let path_str = path.to_string_lossy().to_string();
                    let path_with_ext = if !path_str.ends_with(".sql") {
                        format!("{}.sql", path_str)
                    } else {
                        path_str
                    };

                    match self.generate_and_save_sql(&selected_rules, &category, &path_with_ext) {
                        Ok((success, errors)) => {
                            self.log(&format!("Exported {} rules ({} errors) to {}", success, errors, path_with_ext));
                            nwg::modal_info_message(&self.window, "Export Complete",
                                &format!("Exported {} rules ({} errors) to:\n{}", success, errors, path_with_ext));
                        }
                        Err(e) => {
                            self.log(&format!("ERROR: Export failed: {}", e));
                            nwg::modal_error_message(&self.window, "Export Failed", &format!("{}", e));
                        }
                    }
                }
                Err(e) => {
                    self.log(&format!("ERROR: Failed to get selected file: {}", e));
                }
            }
        } else {
            self.log("Export cancelled");
        }
    }

    fn generate_and_save_sql(&self, rules: &[RuleRow], category: &str, path: &str) -> Result<(usize, usize)> {
        let mut sql = String::with_capacity(rules.len() * 2000); // Pre-allocate
        sql.push_str("-- Generated by Rule Converter GUI\n");
        sql.push_str("-- COMPOSITE template rules for Professional SMART\n\n");

        let mut success_count = 0;
        let mut error_count = 0;
        let mut failed_rules: Vec<(String, String)> = Vec::new(); // (rule_code, error_msg)

        for rule in rules {
            if rule.definition.is_empty() {
                sql.push_str(&format!("-- SKIPPED {}: Empty definition\n\n", rule.filter_number));
                failed_rules.push((rule.filter_number.clone(), "Empty definition".to_string()));
                error_count += 1;
                continue;
            }

            match generate_sql_for_rule(&rule.filter_number, &rule.filter_name, &rule.description, &rule.definition, category) {
                Ok(rule_sql) => {
                    sql.push_str(&rule_sql);
                    sql.push('\n');
                    success_count += 1;
                }
                Err(e) => {
                    let error_msg = e.to_string();
                    sql.push_str(&format!("-- ERROR converting {}: {}\n", rule.filter_number, error_msg));
                    sql.push_str(&format!("-- Definition: {}\n\n", truncate_str(&rule.definition.replace('\n', " "), 200)));
                    failed_rules.push((rule.filter_number.clone(), error_msg));
                    error_count += 1;
                }
            }
        }

        fs::write(path, &sql).map_err(|e| anyhow!("Failed to write file: {}", e))?;

        self.log(&format!("Wrote {} bytes to {}", sql.len(), path));

        // Log failed rules to GUI
        if !failed_rules.is_empty() {
            self.log("");
            self.log(&format!("=== {} RULES FAILED ===", failed_rules.len()));
            for (rule_code, error_msg) in &failed_rules {
                self.log(&format!("  {} : {}", rule_code, error_msg));
            }
            self.log("======================");
        }

        Ok((success_count, error_count))
    }

    fn exit(&self) {
        nwg::stop_thread_dispatch();
    }
}

fn main() {
    nwg::init().expect("Failed to initialize NWG");
    nwg::Font::set_global_family("Segoe UI").expect("Failed to set default font");

    let _app = RuleConverterApp::build_ui(Default::default()).expect("Failed to build UI");
    nwg::dispatch_thread_events();
}
