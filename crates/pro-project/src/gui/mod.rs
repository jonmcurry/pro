mod app;

use eframe::egui;
use eframe::wgpu;

pub fn run() -> anyhow::Result<()> {
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
            .with_inner_size([1100.0, 700.0])
            .with_min_inner_size([900.0, 500.0])
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
        "Professional SMART - Project Database Manager",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::default());
            Ok(Box::new(app::ProjectManagerApp::new(cc)))
        }),
    ) {
        Ok(_) => Ok(()),
        Err(e) => {
            let error_msg = format!(
                "Failed to start GUI: {}\n\n\
                This occurs on Windows Server without GPU/graphics support.\n\n\
                SOLUTION: Use the command-line version instead:\n\n\
                  pro-project.exe --help\n\n\
                Examples:\n\
                  pro-project.exe list              - List all projects\n\
                  pro-project.exe create --name X   - Create project\n\
                  pro-project.exe switch --name X   - Switch project\n\
                  pro-project.exe status            - Show upgrade status\n\n\
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
                    // MB_OK | MB_ICONERROR = 0x10
                    extern "system" {
                        fn MessageBoxW(hwnd: *mut std::ffi::c_void, text: *const u16, caption: *const u16, utype: u32) -> i32;
                    }
                    MessageBoxW(null_mut(), wide_msg.as_ptr(), wide_title.as_ptr(), 0x10);
                }
            }

            Err(anyhow::anyhow!("{}", error_msg))
        }
    }
}

fn load_icon() -> egui::IconData {
    let (icon_rgba, icon_width, icon_height) = {
        let icon_width = 32;
        let icon_height = 32;
        let mut rgba = vec![0u8; icon_width * icon_height * 4];

        let blue = [41, 128, 185, 255];
        let light_blue = [52, 152, 219, 255];
        let white = [255, 255, 255, 255];
        let transparent = [0, 0, 0, 0];

        for i in 0..rgba.len() / 4 {
            let idx = i * 4;
            rgba[idx..idx + 4].copy_from_slice(&transparent);
        }

        let set_pixel = |rgba: &mut Vec<u8>, x: usize, y: usize, color: [u8; 4]| {
            if x < icon_width && y < icon_height {
                let idx = (y * icon_width + x) * 4;
                rgba[idx..idx + 4].copy_from_slice(&color);
            }
        };

        // Draw multiple database cylinders to represent multiple projects
        // First cylinder (back)
        for y in 4..18 {
            for x in 4..18 {
                let dx = (x as f32 - 11.0).abs();
                if dx < 6.0 {
                    set_pixel(&mut rgba, x, y, light_blue);
                }
            }
        }

        // Second cylinder (front)
        for y in 8..22 {
            for x in 10..24 {
                let dx = (x as f32 - 17.0).abs();
                if dx < 6.0 {
                    set_pixel(&mut rgba, x, y, blue);
                }
            }
        }

        // Top ellipse
        for y in 8..11 {
            for x in 10..24 {
                let dx = (x as f32 - 17.0).abs();
                let dy = (y as f32 - 9.5) * 2.5;
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                if dist < 7.0 {
                    set_pixel(&mut rgba, x, y, white);
                }
            }
        }

        // Arrow indicating switch/upgrade
        for i in 0..4 {
            set_pixel(&mut rgba, 24 + i, 14, white);
            set_pixel(&mut rgba, 24 + i, 15, white);
        }
        set_pixel(&mut rgba, 27, 13, white);
        set_pixel(&mut rgba, 28, 14, white);
        set_pixel(&mut rgba, 27, 16, white);

        (rgba, icon_width, icon_height)
    };

    egui::IconData {
        rgba: icon_rgba,
        width: icon_width as u32,
        height: icon_height as u32,
    }
}
