mod app;

use native_windows_gui as nwg;
use nwg::NativeUi;

/// Detach from the console when running in GUI mode
/// Uses FreeConsole() to completely remove the console window, not just hide it
fn detach_console() {
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn FreeConsole() -> i32;
        }

        unsafe {
            FreeConsole();
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    // Detach from the console since we're running in GUI mode
    detach_console();

    // Initialize Native Windows GUI
    nwg::init().map_err(|e| anyhow::anyhow!("Failed to initialize NWG: {}", e))?;

    // Set default font for better appearance
    nwg::Font::set_global_family("Segoe UI")
        .map_err(|e| anyhow::anyhow!("Failed to set font: {}", e))?;

    // Build and run the application
    let app = app::ProjectManagerApp::build_ui(Default::default())
        .map_err(|e| anyhow::anyhow!("Failed to build UI: {}", e))?;

    // Run the event loop
    nwg::dispatch_thread_events();

    // Check if there was an error during execution
    if let Some(error) = app.get_error() {
        return Err(anyhow::anyhow!("{}", error));
    }

    Ok(())
}
