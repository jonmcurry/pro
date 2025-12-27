mod app;

use eframe::egui;

pub fn run() -> anyhow::Result<()> {
    // Use glow (OpenGL) renderer for Windows Server compatibility
    // This provides software rendering fallback when GPU is unavailable
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 700.0])
            .with_min_inner_size([900.0, 500.0])
            .with_icon(load_icon()),
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };

    eframe::run_native(
        "Professional SMART - Project Database Manager",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::default());
            Ok(Box::new(app::ProjectManagerApp::new(cc)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI error: {}", e))
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
