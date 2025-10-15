//! Analytics API endpoints
//!
//! Handlers for querying materialized views (migration 019 - Phase 6)
//! These views are optimized for 10-100x faster queries

use axum::{
    extract::{Query, State},
    response::Json,
};
use super::dashboard::ApiResult;
use super::models::*;
use super::AppState;

// ============================================================================
// Analytics Endpoints (Materialized Views)
// ============================================================================

/// GET /api/v1/analytics/flag-statistics-daily
pub async fn get_flag_statistics_daily(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<FlagStatisticsDaily>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM analytics.flag_statistics_daily WHERE 1=1"
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
        query.push(" AND flag_date >= ");
        query.push_bind(start);
    }

    if let Some(end) = params.end_date {
        query.push(" AND flag_date <= ");
        query.push_bind(end);
    }

    query.push(" ORDER BY flag_date DESC, flag_count DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<FlagStatisticsDaily>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/analytics/encounter-statistics-daily
pub async fn get_encounter_statistics_daily(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<EncounterStatisticsDaily>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM analytics.encounter_statistics_daily WHERE 1=1"
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
        query.push(" AND encounter_date >= ");
        query.push_bind(start);
    }

    if let Some(end) = params.end_date {
        query.push(" AND encounter_date <= ");
        query.push_bind(end);
    }

    query.push(" ORDER BY encounter_date DESC, encounter_count DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<EncounterStatisticsDaily>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/analytics/procedure-statistics
pub async fn get_procedure_statistics(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<ProcedureStatistics>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM analytics.procedure_statistics WHERE 1=1"
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
        .build_query_as::<ProcedureStatistics>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/analytics/provider-performance
pub async fn get_provider_performance(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<ProviderPerformance>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM analytics.provider_performance WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    query.push(" ORDER BY total_rvus DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<ProviderPerformance>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/analytics/payer-statistics
pub async fn get_payer_statistics(
    State(state): State<AppState>,
    Query(params): Query<DashboardQueryParams>,
) -> ApiResult<Json<Vec<PayerStatistics>>> {
    let mut query = sqlx::QueryBuilder::new(
        "SELECT * FROM analytics.payer_statistics WHERE 1=1"
    );

    if let Some(org_id) = params.organization_id {
        query.push(" AND organization_id = ");
        query.push_bind(org_id);
    }

    if let Some(fac_id) = params.facility_id {
        query.push(" AND facility_id = ");
        query.push_bind(fac_id);
    }

    query.push(" ORDER BY denial_rate DESC");

    if let Some(limit) = params.limit {
        query.push(" LIMIT ");
        query.push_bind(limit);
    }

    let results = query
        .build_query_as::<PayerStatistics>()
        .fetch_all(&state.db)
        .await?;

    Ok(Json(results))
}

/// GET /api/v1/analytics/ml-model-performance
pub async fn get_ml_model_performance(
    State(state): State<AppState>,
) -> ApiResult<Json<Vec<MlModelPerformance>>> {
    let results = sqlx::query_as::<_, MlModelPerformance>(
        "SELECT * FROM analytics.ml_model_performance_summary ORDER BY accuracy DESC"
    )
    .fetch_all(&state.db)
    .await?;

    Ok(Json(results))
}
