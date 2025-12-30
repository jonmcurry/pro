mod app;

use eframe::egui;

/// Detach from the console when running in GUI mode
/// Uses FreeConsole() to completely remove the console window
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

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1000.0, 720.0])
            .with_min_inner_size([900.0, 600.0]),
        // Use wgpu backend which can use DirectX on Windows
        hardware_acceleration: eframe::HardwareAcceleration::Preferred,
        ..Default::default()
    };

    eframe::run_native(
        "Professional SMART - Project Database Manager",
        options,
        Box::new(|cc| Ok(Box::new(app::ProjectManagerApp::new(cc)))),
    ).map_err(|e| anyhow::anyhow!("Failed to run GUI: {}", e))
}
