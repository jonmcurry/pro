# Professional SMART API Documentation

## Overview

The Professional SMART API provides REST endpoints for dashboard analytics and real-time WebSocket streaming for claims processing progress.

**Base URL**: `http://127.0.0.1:8080` (configurable via `WEBSOCKET_HOST` environment variable)

**API Version**: v1

## Authentication

Currently, the API does not require authentication. This should be implemented before production deployment.

## Common Query Parameters

Most dashboard endpoints accept these optional query parameters for filtering:

| Parameter | Type | Description |
|-----------|------|-------------|
| `organization_id` | UUID | Filter by organization ID |
| `facility_id` | UUID | Filter by facility ID |
| `start_date` | Date (YYYY-MM-DD) | Filter by start date |
| `end_date` | Date (YYYY-MM-DD) | Filter by end date |
| `limit` | Integer | Limit number of results returned |

**Example**: `/api/v1/dashboard/management-overview?organization_id=123e4567-e89b-12d3-a456-426614174000&start_date=2024-01-01&limit=10`

---

## Health Check

### GET /api/v1/health

Check if the API server is running.

**Response**: `200 OK`
```
OK
```

---

## Dashboard Endpoints

### Management Overview

**GET /api/v1/dashboard/management-overview**

High-level executive metrics by organization, facility, and month.

**Response Fields**:
- `organization_id`, `organization_name`
- `facility_id`, `facility_name`
- `period_month` - Date of the reporting month
- `total_encounters`, `total_service_lines`
- `active_providers`, `active_coders`
- `total_billed_amount`, `avg_claim_amount`
- `total_rvus`, `estimated_medicare_payment`
- `encounters_with_flags`, `total_flag_count`
- `high_severity_flags`, `medium_severity_flags`, `low_severity_flags`
- `flag_rate_percent`
- `total_denials`, `denied_amount`, `denial_rate_percent`

### Claim Status Summary

**GET /api/v1/dashboard/claim-status-summary**

Summary of claims by status (Ready, Submitted, Accepted, Rejected, etc.).

**Response Fields**:
- `organization_id`, `facility_id`
- `claim_status`
- `encounter_count`
- `total_billed_amount`, `avg_billed_amount`

### Coder Performance

**GET /api/v1/dashboard/coder-performance**

Coder productivity and accuracy metrics (30-day rolling window).

**Response Fields**:
- `coder_id`, `coder_name`, `organization_id`
- `encounters_coded`, `service_lines_coded`
- `work_rvus`, `total_rvus`
- `audits_conducted`, `audits_passed`, `audit_accuracy_rate`
- `critical_errors`, `major_errors`, `minor_errors`
- `total_overpayment`, `total_underpayment`
- `flags_generated`, `flags_accepted`
- `avg_encounters_per_day`

### Provider Documentation Accuracy

**GET /api/v1/dashboard/provider-documentation-accuracy**

Provider documentation quality metrics (90-day rolling window).

**Response Fields**:
- `provider_id`, `provider_name`
- `organization_id`, `facility_id`
- `encounters_documented`, `service_lines`
- `audits_reviewed`, `documentation_accuracy_rate`
- `overcoding_instances`, `undercoding_instances`, `unsupported_instances`
- `total_overpayment_risk`, `total_underpayment_risk`

### Flags by Category

**GET /api/v1/dashboard/flags-by-category**

Flag statistics by category, issue type, and severity.

**Response Fields**:
- `organization_id`, `facility_id`
- `flag_category`, `issue_type`, `severity_level`
- `flag_count`, `open_flags`, `resolved_flags`
- `accepted_flags`, `rejected_flags`
- `resolution_rate_percent`
- `avg_resolution_time_hours`

### Service Line Flags Detail

**GET /api/v1/dashboard/service-line-flags-detail**

Detailed view of individual service line flags.

**Default Limit**: 1000 records

**Response Fields**:
- `flag_id`, `encounter_id`, `service_line_number`
- `flag_category`, `issue_type`, `severity_level`
- `flag_description`, `proposed_correction`
- `procedure_code`, `charged_amount`
- `flag_status`, `resolution_notes`
- `coder_id`, `coder_name`
- `provider_id`, `provider_name`, `facility_name`

### Denial by Payer

**GET /api/v1/dashboard/denial-by-payer**

Denial analysis by payer and month.

**Response Fields**:
- `organization_id`, `facility_id`
- `payer_id`, `payer_name`, `period_month`
- `denial_count`, `denied_amount`
- `total_billed_amount`, `denial_rate_percent`
- `coding_error_denials`, `documentation_denials`
- `authorization_denials`, `timely_filing_denials`, `other_denials`
- `preventable_denials`
- `appeals_filed`, `appeals_overturned`, `appeal_success_rate`

### Denial by Reason

**GET /api/v1/dashboard/denial-by-reason**

Denial analysis by CARC reason code.

**Response Fields**:
- `organization_id`, `facility_id`
- `denial_reason_code`, `denial_reason_description`
- `denial_count`, `denied_amount`
- `preventable_count`, `overturned_count`, `written_off_count`

### Procedure Volume

**GET /api/v1/dashboard/procedure-volume**

Procedure volume and performance analysis.

**Response Fields**:
- `organization_id`, `facility_id`
- `procedure_code`, `procedure_description`
- `procedure_count`, `total_units`, `total_charges`
- `total_rvus`, `estimated_payment`
- `flag_count`, `flag_rate_percent`

### Provider Productivity

**GET /api/v1/dashboard/provider-productivity**

Provider RVU and productivity analysis by month.

**Response Fields**:
- `provider_id`, `provider_name`
- `organization_id`, `facility_id`, `period_month`
- `encounter_count`, `service_line_count`
- `total_charges`, `total_work_rvus`, `total_rvus`
- `estimated_collections`, `avg_rvus_per_encounter`
- `em_visits`, `non_em_procedures`

### Audit Assignment Status

**GET /api/v1/dashboard/audit-assignment-status**

Audit assignment progress tracking.

**Response Fields**:
- `assignment_id`, `organization_id`, `facility_id`
- `audit_type`, `sample_size`
- `completed_count`, `completion_percent`
- `errors_found`, `error_rate`, `flags_generated`
- `total_overpayment`, `total_underpayment`
- `reviewer_id`, `reviewer_name`
- `assigned_date`, `due_date`
- `days_in_progress`, `days_until_due`

### Reimbursement Analysis

**GET /api/v1/dashboard/reimbursement-analysis**

Comprehensive reimbursement and financial analysis by month.

**Response Fields**:
- `organization_id`, `facility_id`, `period_month`
- `encounter_count`, `service_line_count`
- `total_charges`, `total_rvus`, `rvu_based_estimate`
- `charge_to_rvu_ratio`
- `denial_count`, `denial_amount`, `net_expected_payment`

---

## Analytics Endpoints (Materialized Views)

These endpoints query pre-aggregated materialized views optimized for 10-100x faster performance.

**Note**: Materialized views should be refreshed daily using:
```sql
SELECT analytics.refresh_all_views();
```

### Flag Statistics Daily

**GET /api/v1/analytics/flag-statistics-daily**

Daily flag aggregations with financial impact (90-day window).

**Response Fields**:
- `organization_id`, `facility_id`, `flag_date`
- `flag_category`, `severity_level`
- `flag_count`, `resolved_count`, `accepted_count`
- `median_resolution_hours`, `total_financial_impact`

### Encounter Statistics Daily

**GET /api/v1/analytics/encounter-statistics-daily**

Daily encounter volume and financial statistics (90-day window).

**Response Fields**:
- `organization_id`, `facility_id`, `encounter_date`
- `claim_status`, `payer_id`
- `encounter_count`, `total_charges`, `total_rvus`
- `service_line_count`

### Procedure Statistics

**GET /api/v1/analytics/procedure-statistics**

Procedure code usage with flag rates (90-day window, minimum 5 occurrences).

**Response Fields**:
- `organization_id`, `facility_id`
- `procedure_code`, `procedure_count`
- `total_charges`, `total_rvus`
- `flag_count`, `flag_rate`
- `common_modifiers` - Array of commonly used modifiers

### Provider Performance

**GET /api/v1/analytics/provider-performance**

Provider activity and quality metrics (90-day window, minimum 5 encounters).

**Response Fields**:
- `organization_id`, `facility_id`, `provider_id`
- `encounter_count`, `total_charges`, `total_rvus`
- `flag_count`, `flags_per_encounter`

### Payer Statistics

**GET /api/v1/analytics/payer-statistics**

Payer performance with denial rates (90-day window, minimum 5 encounters).

**Response Fields**:
- `organization_id`, `facility_id`, `payer_id`
- `encounter_count`, `total_charges`
- `denial_count`, `denial_rate`
- `top_procedures` - Array of most common procedure codes

### ML Model Performance

**GET /api/v1/analytics/ml-model-performance**

ML model accuracy and prediction statistics (30-day window).

**Response Fields**:
- `model_name`
- `prediction_count`, `correct_predictions`
- `accuracy`, `avg_confidence_score`

---

## Queue Monitoring Endpoints

### Queue Health

**GET /api/v1/queue/health**

Real-time queue health by facility (last 24 hours).

**Response Fields**:
- `facility_id`, `facility_name`
- `queued_count`, `processing_count`, `completed_count`, `failed_count`
- `oldest_queued`, `newest_queued`
- `avg_processing_time_seconds`, `max_processing_time_seconds`

### Queue Statistics

**GET /api/v1/queue/statistics**

Hourly file processing queue performance (last 7 days).

**Default Limit**: 168 (7 days × 24 hours)

**Response Fields**:
- `facility_id`, `facility_name`, `hour`
- `files_queued`, `files_completed`, `files_failed`
- `completion_rate`
- `avg_queue_wait_seconds`, `avg_processing_seconds`

### FIFO Violations

**GET /api/v1/queue/fifo-violations**

Detects claims processed out of service date order (last 30 days).

**Default Limit**: 100

**Response Fields**:
- `organization_id`, `facility_id`
- `earlier_encounter_id`, `earlier_service_date`, `earlier_created_at`
- `later_encounter_id`, `later_service_date`, `later_created_at`
- `time_gap_hours`

---

## WebSocket Endpoint

### Real-Time Processing Stream

**WS /api/v1/processing/stream/:queue_id**

Connect to receive real-time updates on claims processing progress.

**Parameters**:
- `queue_id` - UUID of the processing queue to monitor

**Connection**:
```javascript
const ws = new WebSocket('ws://127.0.0.1:8080/api/v1/processing/stream/YOUR_QUEUE_ID');
```

**Message Types** (JSON):

1. **Connected**
```json
{
  "type": "connected",
  "queue_id": "uuid",
  "message": "WebSocket connection established"
}
```

2. **Started**
```json
{
  "type": "started",
  "queue_id": "uuid",
  "total_claims": 1000
}
```

3. **Progress**
```json
{
  "type": "progress",
  "queue_id": "uuid",
  "processed": 250,
  "total": 1000,
  "percent": 25.0
}
```

4. **Claim Processed**
```json
{
  "type": "claim_processed",
  "queue_id": "uuid",
  "claim_id": "uuid"
}
```

5. **Claim Failed**
```json
{
  "type": "claim_failed",
  "queue_id": "uuid",
  "claim_id": "uuid",
  "error": "Error message"
}
```

6. **Completed**
```json
{
  "type": "completed",
  "queue_id": "uuid",
  "total_processed": 1000,
  "duration_seconds": 15
}
```

7. **Failed**
```json
{
  "type": "failed",
  "queue_id": "uuid",
  "error": "Error message"
}
```

---

## Error Responses

All endpoints return standard HTTP status codes:

| Code | Description |
|------|-------------|
| 200 | Success |
| 404 | Resource not found |
| 500 | Internal server error |

**Error Response Format**:
```json
{
  "error": "Error message describing what went wrong"
}
```

---

## Configuration

Set these environment variables in your `.env` file:

```env
# Enable/disable API server
STREAMING_ENABLE_WEBSOCKET=true

# API server address
WEBSOCKET_HOST=127.0.0.1:8080

# Database connection (required for REST endpoints)
DATABASE_URL=postgres://postgres:password@localhost:5432/professional_smart
```

---

## Usage Examples

### Fetch Management Overview (cURL)

```bash
curl "http://127.0.0.1:8080/api/v1/dashboard/management-overview?limit=10"
```

### Fetch Coder Performance with Filters (cURL)

```bash
curl "http://127.0.0.1:8080/api/v1/dashboard/coder-performance?organization_id=123e4567-e89b-12d3-a456-426614174000&limit=20"
```

### WebSocket Connection (JavaScript)

```javascript
const ws = new WebSocket('ws://127.0.0.1:8080/api/v1/processing/stream/YOUR_QUEUE_ID');

ws.onopen = () => {
  console.log('Connected to processing stream');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Progress update:', data);
};

ws.onerror = (error) => {
  console.error('WebSocket error:', error);
};

ws.onclose = () => {
  console.log('Connection closed');
};
```

### React Hook Example

```typescript
import { useEffect, useState } from 'react';

function useProcessingStream(queueId: string) {
  const [progress, setProgress] = useState({ processed: 0, total: 0 });

  useEffect(() => {
    const ws = new WebSocket(`ws://127.0.0.1:8080/api/v1/processing/stream/${queueId}`);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'progress') {
        setProgress({ processed: data.processed, total: data.total });
      }
    };

    return () => ws.close();
  }, [queueId]);

  return progress;
}
```

---

## Performance Considerations

1. **Materialized Views**: Analytics endpoints use materialized views that must be refreshed daily for accurate data
2. **Query Limits**: Use the `limit` parameter to control response size
3. **Filtering**: Always filter by `organization_id` or `facility_id` when possible to improve query performance
4. **Connection Pooling**: The API uses a connection pool with 50 max connections
5. **Statement Caching**: Prepared statements are cached for improved performance on repeated queries

---

## TODO / Future Enhancements

- [ ] Add authentication middleware (JWT or API keys)
- [ ] Add rate limiting
- [ ] Add CORS configuration for frontend domains
- [ ] Add OpenAPI/Swagger documentation
- [ ] Add pagination support for large result sets
- [ ] Add response compression (gzip)
- [ ] Add request validation middleware
- [ ] Add metrics/monitoring endpoints (Prometheus format)
- [ ] Add bulk export endpoints (CSV, Excel)
- [ ] Add filtering by date ranges in all relevant endpoints
