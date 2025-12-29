//! Dashboard API endpoints
//!
//! Handlers for querying dashboard views (migration 013 and 015)

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Json, Response},
};
use serde_json::json;

use super::models::*;
use super::AppState;

// ============================================================================
// Error Handling
// ============================================================================

#[derive(Debug)]
pub enum ApiError {
    DatabaseError(sqlx::Error),
    NotFound,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::DatabaseError(e) => {
                tracing::error!("Database error: {:?}", e);
                (StatusCode::INTERNAL_SERVER_ERROR, "Database error occurred")
            }
            ApiError::NotFound => (StatusCode::NOT_FOUND, "Resource not found"),
        };

        let body = Json(json!({
            "error": message
        }));

        (status, body).into_response()
    }
}

impl From<sqlx::Error> for ApiError {
    fn from(error: sqlx::Error) -> Self {
        ApiError::DatabaseError(error)
    }
}

pub type ApiResult<T> = Result<T, ApiError>;

// ============================================================================
// Dashboard Endpoints
// ============================================================================

/// GET /api/v1/dashboard/management-overview
pub async fn get_management_overview(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<ManagementOverview>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_management_overview WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    if let Some(start) = params.start_date {
        query.push(" AND period_month >= ");
        query.push_bind(start);
    }

    if let Some(end) = params.end_date {
        query.push(" AND period_month <= ");
        query.push_bind(end);
    }

    query.push(" ORDER BY period_month DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<ManagementOverview>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/claim-status-summary
pub async fn get_claim_status_summary(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<ClaimStatusSummary>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_claim_status_summary WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    let results = query
        .build_query_as::<ClaimStatusSummary>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/coder-performance
pub async fn get_coder_performance(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<CoderPerformance>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_coder_performance WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    query.push(" ORDER BY encounters_coded DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<CoderPerformance>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/provider-documentation-accuracy
pub async fn get_provider_documentation_accuracy(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<ProviderDocumentationAccuracy>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_provider_documentation_accuracy WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    query.push(" ORDER BY encounters_documented DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<ProviderDocumentationAccuracy>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/flags-by-category
pub async fn get_flags_by_category(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<FlagsByCategory>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_flags_by_category WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    query.push(" ORDER BY flag_count DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<FlagsByCategory>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/service-line-flags-detail
pub async fn get_service_line_flags_detail(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<ServiceLineFlagsDetail>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_service_line_flags_detail WHERE 1=1"
    );

    if let Some(_fac_id) = params.facility_id {
        query.push(" AND facility_name IS NOT NULL"); // Filter exists
    }

    query.push(" ORDER BY charged_amount DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    } else {
        query.push(" LIMIT 1000"); // Default limit for detail view
    }

    let results = query
        .build_query_as::<ServiceLineFlagsDetail>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/denial-by-payer
pub async fn get_denial_by_payer(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<DenialByPayer>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_denial_by_payer WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    if let Some(start) = params.start_date {
        query.push(" AND period_month >= ");
        query.push_bind(start);
    }

    if let Some(end) = params.end_date {
        query.push(" AND period_month <= ");
        query.push_bind(end);
    }

    query.push(" ORDER BY denial_rate_percent DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<DenialByPayer>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/denial-by-reason
pub async fn get_denial_by_reason(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<DenialByReason>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_denial_by_reason WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    query.push(" ORDER BY denied_amount DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<DenialByReason>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/procedure-volume
pub async fn get_procedure_volume(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<ProcedureVolume>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_procedure_volume WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    query.push(" ORDER BY procedure_count DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<ProcedureVolume>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/provider-productivity
pub async fn get_provider_productivity(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<ProviderProductivity>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_provider_productivity WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    if let Some(start) = params.start_date {
        query.push(" AND period_month >= ");
        query.push_bind(start);
    }

    if let Some(end) = params.end_date {
        query.push(" AND period_month <= ");
        query.push_bind(end);
    }

    query.push(" ORDER BY total_rvus DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<ProviderProductivity>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/audit-assignment-status
pub async fn get_audit_assignment_status(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<AuditAssignmentStatus>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_audit_assignment_status WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    query.push(" ORDER BY due_date ASC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<AuditAssignmentStatus>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/dashboard/reimbursement-analysis
pub async fn get_reimbursement_analysis(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<ReimbursementAnalysis>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_reimbursement_analysis WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    if let Some(start) = params.start_date {
        query.push(" AND period_month >= ");
        query.push_bind(start);
    }

    if let Some(end) = params.end_date {
        query.push(" AND period_month <= ");
        query.push_bind(end);
    }

    query.push(" ORDER BY period_month DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<ReimbursementAnalysis>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

// ============================================================================
// Queue Monitoring Endpoints
// ============================================================================

/// GET /api/v1/queue/health
pub async fn get_queue_health(
    State(state): State<AppState>,
) -> ApiResult<Json<Vec<QueueHealth>>> {
    let results = sqlx::query_as::<_, QueueHealth>(
        "SELECT * FROM staging.v_queue_health ORDER BY facility_name"
    )
    .fetch_all(&state.db)
    .await?;

    Ok(Json(results))
}

/// GET /api/v1/queue/statistics
pub async fn get_queue_statistics(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<QueueStatistics>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM staging.v_queue_statistics WHERE 1=1"
    );

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    query.push(" ORDER BY hour DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    } else {
        query.push(" LIMIT 168"); // Default 7 days * 24 hours
    }

    let results = query
        .build_query_as::<QueueStatistics>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/queue/fifo-violations
pub async fn get_fifo_violations(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<FifoViolation>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM claims.v_fifo_violations WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    query.push(" ORDER BY time_gap_hours DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    } else {
        query.push(" LIMIT 100"); // Default limit
    }

    let results = query
        .build_query_as::<FifoViolation>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}
