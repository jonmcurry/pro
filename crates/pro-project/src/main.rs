#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod cli;
mod gui;
mod services;

use clap::Parser;
use cli::Cli;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

fn main() -> anyhow::Result<()> {
    // Initialize tracing for logging
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "pro_project=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse command line arguments
    let cli = Cli::parse();

    // Check if GUI mode is requested
    if cli.gui || matches!(cli.command, Some(cli::Commands::Gui)) {
        return gui::run();
    }

    // Run CLI mode
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(cli::run(cli))
}
