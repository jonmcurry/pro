#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;

use eframe::egui;

fn main() -> Result<(), eframe::Error> {
    // Use wgpu renderer for Windows Server 2019 compatibility
    // wgpu can use DirectX 12 WARP (software renderer) when GPU is unavailable
    // This works on headless Windows Server environments where OpenGL is not available
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([900.0, 700.0])
            .with_min_inner_size([800.0, 600.0])
            .with_icon(load_icon()),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };

    eframe::run_native(
        "Professional SMART - Master Data Loader",
        options,
        Box::new(|cc| {
            // Use system theme
            cc.egui_ctx.set_visuals(egui::Visuals::default());
            Ok(Box::new(app::DataLoaderApp::new(cc)))
        }),
    )
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
