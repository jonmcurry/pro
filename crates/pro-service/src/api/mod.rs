//! API module for REST endpoints and WebSocket server
//!
//! This module provides HTTP REST APIs and WebSocket streaming for the
//! Professional SMART claims processing system.

pub mod dashboard;
pub mod analytics;
pub mod models;

use axum::{
    routing::get,
    Router,
};
use sqlx::PgPool;
use std::sync::Arc;
use crate::websocket::{ws_handler, WebSocketState};

/// Shared application state for REST endpoints
#[derive(Clone)]
pub struct AppState {
    pub db: PgPool,
}

/// Create the Axum application router with all endpoints
pub fn create_app(db: PgPool) -> Router {
    let ws_state = Arc::new(WebSocketState::new());
    let app_state = AppState { db };

    // Create REST API router with database pool
    let rest_router = Router::new()
        // Dashboard endpoints
        .route("/api/v1/dashboard/management-overview", get(dashboard::get_management_overview))
        .route("/api/v1/dashboard/claim-status-summary", get(dashboard::get_claim_status_summary))
        .route("/api/v1/dashboard/coder-performance", get(dashboard::get_coder_performance))
        .route("/api/v1/dashboard/provider-documentation-accuracy", get(dashboard::get_provider_documentation_accuracy))
        .route("/api/v1/dashboard/flags-by-category", get(dashboard::get_flags_by_category))
        .route("/api/v1/dashboard/service-line-flags-detail", get(dashboard::get_service_line_flags_detail))
        .route("/api/v1/dashboard/denial-by-payer", get(dashboard::get_denial_by_payer))
        .route("/api/v1/dashboard/denial-by-reason", get(dashboard::get_denial_by_reason))
        .route("/api/v1/dashboard/procedure-volume", get(dashboard::get_procedure_volume))
        .route("/api/v1/dashboard/provider-productivity", get(dashboard::get_provider_productivity))
        .route("/api/v1/dashboard/audit-assignment-status", get(dashboard::get_audit_assignment_status))
        .route("/api/v1/dashboard/reimbursement-analysis", get(dashboard::get_reimbursement_analysis))
        // Analytics endpoints (materialized views)
        .route("/api/v1/analytics/flag-statistics-daily", get(analytics::get_flag_statistics_daily))
        .route("/api/v1/analytics/encounter-statistics-daily", get(analytics::get_encounter_statistics_daily))
        .route("/api/v1/analytics/procedure-statistics", get(analytics::get_procedure_statistics))
        .route("/api/v1/analytics/provider-performance", get(analytics::get_provider_performance))
        .route("/api/v1/analytics/payer-statistics", get(analytics::get_payer_statistics))
        .route("/api/v1/analytics/ml-model-performance", get(analytics::get_ml_model_performance))
        // Queue monitoring endpoints
        .route("/api/v1/queue/health", get(dashboard::get_queue_health))
        .route("/api/v1/queue/statistics", get(dashboard::get_queue_statistics))
        .route("/api/v1/queue/fifo-violations", get(dashboard::get_fifo_violations))
        // Health check
        .route("/api/v1/health", get(health_check))
        .with_state(app_state);

    // Create WebSocket router with WebSocketState
    let ws_router = Router::new()
        .route("/api/v1/processing/stream/:queue_id", get(ws_handler))
        .with_state(ws_state);

    // Merge both routers
    Router::new()
        .merge(rest_router)
        .merge(ws_router)
}

/// Start the API server
pub async fn serve(addr: &str, db: PgPool) -> Result<(), Box<dyn std::error::Error>> {
    let app = create_app(db);
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!("API server listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "OK"
}
