//! WebSocket Support for Real-time Progress Updates
//!
//! PHASE 5: Provides WebSocket connections for streaming progress updates

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Path, State,
    },
    response::Response,
};
use futures::{sink::SinkExt, stream::StreamExt};
use pro_worker::ProgressEvent;
use serde_json;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{error, info, warn};


/// Shared state for WebSocket connections
#[derive(Clone)]
pub struct WebSocketState {
    /// Broadcast channel for progress events
    pub progress_broadcaster: broadcast::Sender<ProgressEvent>,
}

impl WebSocketState {
    /// Create new WebSocket state
    pub fn new() -> Self {
        // Create broadcast channel with capacity for 1000 messages
        let (tx, _rx) = broadcast::channel(1000);
        Self {
            progress_broadcaster: tx,
        }
    }

    /// Get a subscriber to progress events
    pub fn subscribe(&self) -> broadcast::Receiver<ProgressEvent> {
        self.progress_broadcaster.subscribe()
    }

    /// Get the broadcaster (for ProgressTracker)
    /// Reserved for future progress tracking integration
    #[allow(dead_code)]
    pub fn broadcaster(&self) -> broadcast::Sender<ProgressEvent> {
        self.progress_broadcaster.clone()
    }
}

impl Default for WebSocketState {
    fn default() -> Self {
        Self::new()
    }
}

/// WebSocket handler for streaming progress updates
///
/// Endpoint: GET /api/v1/processing/stream/{queue_id}
/// Upgrade: websocket
///
/// This handler upgrades the HTTP connection to a WebSocket and streams
/// real-time progress events for the specified queue_id.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Path(queue_id): Path<i64>,
    State(state): State<Arc<WebSocketState>>,
) -> Response {
    info!("WebSocket connection request for queue_id: {}", queue_id);

    ws.on_upgrade(move |socket| handle_socket(socket, queue_id, state))
}

/// Handle WebSocket connection
async fn handle_socket(socket: WebSocket, queue_id: i64, state: Arc<WebSocketState>) {
    info!("WebSocket connected for queue_id: {}", queue_id);

    let (mut sender, mut receiver) = socket.split();

    // Subscribe to progress events
    let mut progress_rx = state.subscribe();

    // Send initial connection confirmation
    let welcome_msg = serde_json::json!({
        "type": "connected",
        "queue_id": queue_id,
        "message": "WebSocket connection established"
    });

    if let Ok(json) = serde_json::to_string(&welcome_msg) {
        if sender.send(Message::Text(json)).await.is_err() {
            error!("Failed to send welcome message");
            return;
        }
    }

    // Spawn task to handle outgoing messages (progress events)
    let mut send_task = tokio::spawn(async move {
        while let Ok(event) = progress_rx.recv().await {
            // Filter events for this specific queue_id
            let matches_queue = match &event {
                ProgressEvent::Started { queue_id: qid, .. } => qid == &queue_id,
                ProgressEvent::Progress { queue_id: qid, .. } => qid == &queue_id,
                ProgressEvent::ClaimProcessed { queue_id: qid, .. } => qid == &queue_id,
                ProgressEvent::ClaimFailed { queue_id: qid, .. } => qid == &queue_id,
                ProgressEvent::Completed { queue_id: qid, .. } => qid == &queue_id,
                ProgressEvent::Failed { queue_id: qid, .. } => qid == &queue_id,
            };

            if !matches_queue {
                continue;
            }

            // Serialize and send event
            match serde_json::to_string(&event) {
                Ok(json) => {
                    if sender.send(Message::Text(json)).await.is_err() {
                        warn!("Failed to send progress event, client disconnected");
                        break;
                    }
                }
                Err(e) => {
                    error!("Failed to serialize progress event: {}", e);
                }
            }

            // If this is a completion or failure event, close the connection
            if matches!(
                event,
                ProgressEvent::Completed { .. } | ProgressEvent::Failed { .. }
            ) {
                info!("Processing completed for queue_id: {}, closing WebSocket", queue_id);
                break;
            }
        }
    });

    // Spawn task to handle incoming messages (ping/pong, close)
    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = receiver.next().await {
            match msg {
                Message::Text(text) => {
                    // Handle client messages (e.g., ping request)
                    if text == "ping" {
                        // Client sent ping, we'll respond with pong via send_task
                        // For now, just log it
                        info!("Received ping from client for queue_id: {}", queue_id);
                    }
                }
                Message::Close(_) => {
                    info!("Client closed WebSocket for queue_id: {}", queue_id);
                    break;
                }
                Message::Ping(_data) => {
                    // Axum automatically handles pong responses
                    info!("Received ping with data for queue_id: {}", queue_id);
                }
                Message::Pong(_) => {
                    // Pong received, connection is alive
                }
                _ => {
                    // Binary messages not supported
                    warn!("Unsupported WebSocket message type");
                }
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = (&mut send_task) => {
            recv_task.abort();
        },
        _ = (&mut recv_task) => {
            send_task.abort();
        },
    }

    info!("WebSocket disconnected for queue_id: {}", queue_id);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_websocket_state_creation() {
        let state = WebSocketState::new();
        let _subscriber = state.subscribe();
        // State should be created successfully
    }

    #[tokio::test]
    async fn test_broadcast_event() {
        use chrono::Utc;

        let state = WebSocketState::new();
        let mut rx = state.subscribe();

        let test_event = ProgressEvent::Started {
            queue_id: 1,
            progress_id: 1,
            total_claims: 100,
            started_at: Utc::now(),
        };

        // Send event
        state.progress_broadcaster.send(test_event.clone()).unwrap();

        // Receive event
        let received = rx.recv().await.unwrap();
        match received {
            ProgressEvent::Started { total_claims, .. } => {
                assert_eq!(total_claims, 100);
            }
            _ => panic!("Expected Started event"),
        }
    }
}
