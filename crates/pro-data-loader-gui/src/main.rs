#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;

use eframe::egui;
use eframe::wgpu;

fn main() -> Result<(), eframe::Error> {
    // Windows Server 2019 compatibility: Force DX12 backend which has WARP software renderer
    // WARP (Windows Advanced Rasterization Platform) is available on Windows Server 2019
    // Reference: https://learn.microsoft.com/en-us/windows/win32/direct3darticles/directx-warp
    if std::env::var("WGPU_BACKEND").is_err() {
        std::env::set_var("WGPU_BACKEND", "dx12");
    }

    // Low power preference helps select software adapters
    if std::env::var("WGPU_POWER_PREF").is_err() {
        std::env::set_var("WGPU_POWER_PREF", "low");
    }

    // Allow non-compliant adapters (software renderers like WARP)
    if std::env::var("WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER").is_err() {
        std::env::set_var("WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER", "1");
    }

    // Configure wgpu to use DX12 with WARP fallback
    let wgpu_options = eframe::egui_wgpu::WgpuConfiguration {
        // Only use DX12 backend - has WARP software renderer on Windows Server
        supported_backends: wgpu::Backends::DX12,
        // Use low power preference to help select WARP when no GPU available
        power_preference: wgpu::PowerPreference::LowPower,
        ..Default::default()
    };

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_min_inner_size([800.0, 600.0])
            .with_icon(load_icon()),
        renderer: eframe::Renderer::Wgpu,
        wgpu_options,
        // Disable multisampling - required for software renderers like WARP
        // Reference: https://github.com/emilk/egui/issues/957
        multisampling: 0,
        depth_buffer: 0,
        ..Default::default()
    };

    match eframe::run_native(
        "Professional SMART - Master Data Loader",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::default());
            Ok(Box::new(app::DataLoaderApp::new(cc)))
        }),
    ) {
        Ok(_) => Ok(()),
        Err(e) => {
            let error_msg = format!(
                "Failed to start GUI: {}\n\n\
                This occurs on Windows Server without GPU/graphics support.\n\n\
                SOLUTION: Use the command-line version instead:\n\n\
                  pro-data-loader.exe --help\n\n\
                Example:\n\
                  pro-data-loader.exe --csv-dir C:\\data\\master\n\n\
                The CLI version has full functionality and works on all Windows systems.",
                e
            );

            // Try to show a message box on Windows
            #[cfg(windows)]
            {
                use std::ffi::OsStr;
                use std::os::windows::ffi::OsStrExt;
                use std::ptr::null_mut;

                let wide_msg: Vec<u16> = OsStr::new(&error_msg)
                    .encode_wide()
                    .chain(std::iter::once(0))
                    .collect();
                let wide_title: Vec<u16> = OsStr::new("Professional SMART - GUI Error")
                    .encode_wide()
                    .chain(std::iter::once(0))
                    .collect();

                unsafe {
                    extern "system" {
                        fn MessageBoxW(hwnd: *mut std::ffi::c_void, text: *const u16, caption: *const u16, utype: u32) -> i32;
                    }
                    MessageBoxW(null_mut(), wide_msg.as_ptr(), wide_title.as_ptr(), 0x10);
                }
            }

            Err(e)
        }
    }
}

fn load_icon() -> egui::IconData {
    // Create a professional icon with document/database theme
    let (icon_rgba, icon_width, icon_height) = {
        let icon_width = 32;
        let icon_height = 32;
        let mut rgba = vec![0u8; icon_width * icon_height * 4];

        // Define colors
        let blue = [41, 128, 185, 255];      // Professional blue
        let light_blue = [52, 152, 219, 255]; // Lighter blue
        let white = [255, 255, 255, 255];
        let transparent = [0, 0, 0, 0];

        // Fill with transparent background
        for i in 0..rgba.len() / 4 {
            let idx = i * 4;
            rgba[idx..idx + 4].copy_from_slice(&transparent);
        }

        // Helper to set pixel
        let set_pixel = |rgba: &mut Vec<u8>, x: usize, y: usize, color: [u8; 4]| {
            if x < icon_width && y < icon_height {
                let idx = (y * icon_width + x) * 4;
                rgba[idx..idx + 4].copy_from_slice(&color);
            }
        };

        // Draw a database icon (cylinder stack)
        // Top ellipse
        for y in 6..10 {
            for x in 8..24 {
                let dx = (x as f32 - 16.0).abs();
                let dy = (y as f32 - 8.0) * 2.0;
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                if dist < 8.5 {
                    set_pixel(&mut rgba, x, y, if y < 8 { light_blue } else { blue });
                }
            }
        }

        // Middle section (cylinder body)
        for y in 9..18 {
            for x in 8..24 {
                let dx = (x as f32 - 16.0).abs();
                if dx < 8.0 {
                    set_pixel(&mut rgba, x, y, blue);
                }
            }
        }

        // Middle ellipse line
        for x in 8..24 {
            let dx = (x as f32 - 16.0).abs();
            if dx < 8.0 {
                set_pixel(&mut rgba, x, 14, light_blue);
            }
        }

        // Bottom ellipse
        for y in 17..21 {
            for x in 8..24 {
                let dx = (x as f32 - 16.0).abs();
                let dy = (y as f32 - 19.0) * 2.0;
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                if dist < 8.5 {
                    set_pixel(&mut rgba, x, y, if y > 18 { blue } else { light_blue });
                }
            }
        }

        // Add CSV/document overlay in corner
        // Small document shape
        for y in 22..28 {
            for x in 18..26 {
                set_pixel(&mut rgba, x, y, white);
            }
        }
        // Document fold
        for i in 0..3 {
            set_pixel(&mut rgba, 25 - i, 22 + i, light_blue);
        }
        // Document lines
        set_pixel(&mut rgba, 20, 24, blue);
        set_pixel(&mut rgba, 21, 24, blue);
        set_pixel(&mut rgba, 22, 24, blue);
        set_pixel(&mut rgba, 20, 26, blue);
        set_pixel(&mut rgba, 21, 26, blue);

        (rgba, icon_width, icon_height)
    };

    egui::IconData {
        rgba: icon_rgba,
        width: icon_width as u32,
        height: icon_height as u32,
    }
}
