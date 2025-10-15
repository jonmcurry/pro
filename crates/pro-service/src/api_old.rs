//! Axum API server for WebSocket and REST endpoints
//!
//! PHASE 5: Provides HTTP/WebSocket server for real-time progress updates

use axum::{
    routing::get,
    Router,
};
use std::sync::Arc;
use crate::websocket::{ws_handler, WebSocketState};

/// Create the Axum application router
pub fn create_app() -> Router {
    let ws_state = Arc::new(WebSocketState::new());

    Router::new()
        .route("/api/v1/processing/stream/:queue_id", get(ws_handler))
        .with_state(ws_state)
}

/// Start the WebSocket/API server
pub async fn serve(addr: &str) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_app();
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!("WebSocket server listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
