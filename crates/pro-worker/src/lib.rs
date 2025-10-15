// Ingestion worker for Professional SMART
//
// This crate provides the file processing pipeline that integrates:
// - EDI 837p parser
// - CSV parser
// - Data validation and deduplication
// - Rules engine for flagging
// - RVU-based reimbursement calculation
// - Database persistence
//
// # Processing Pipeline
//
// 1. File ingestion (detect format, calculate hash)
// 2. Parse file (EDI or CSV)
// 3. Validate data (business rules, duplicates)
// 4. Execute rules engine (create flags)
// 5. Calculate RVU payments
// 6. Persist to database
// 7. Update processing statistics

pub mod pipeline;
pub mod queue_manager;
pub mod types;
pub mod progress; // PHASE 5: Real-time progress tracking

// Re-export commonly used items
pub use pipeline::IngestionPipeline;
pub use queue_manager::{QueueManager, QueueStatus, QueuedFile};
pub use types::{
    ClaimProcessingResult, FileFormat, FlagSeverityCount, IngestionJob, ProcessingStats,
    ProcessingStatus,
};
pub use progress::{ProgressTracker, ProgressEvent, ProgressSnapshot}; // PHASE 5
pub use pro_common::{Error, Result};
