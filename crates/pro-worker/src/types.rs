// Core types for the ingestion worker

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// File format for ingestion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileFormat {
    /// 837p EDI format
    Edi837p,
    /// CSV format (various EHR exports)
    Csv,
}

impl FileFormat {
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "edi" | "837" | "x12" | "txt" => Some(FileFormat::Edi837p),
            "csv" => Some(FileFormat::Csv),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            FileFormat::Edi837p => "837p_EDI",
            FileFormat::Csv => "CSV",
        }
    }
}

/// Processing status for a file
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingStatus {
    /// File received, queued for processing
    Queued,
    /// File is currently being processed
    Processing,
    /// File processed successfully
    Completed,
    /// File processing failed
    Failed,
    /// File partially processed (some records succeeded, some failed)
    Partial,
}

impl ProcessingStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProcessingStatus::Queued => "QUEUED",
            ProcessingStatus::Processing => "PROCESSING",
            ProcessingStatus::Completed => "COMPLETED",
            ProcessingStatus::Failed => "FAILED",
            ProcessingStatus::Partial => "PARTIAL",
        }
    }
}

/// File ingestion job
#[derive(Debug, Clone)]
pub struct IngestionJob {
    /// Unique job ID
    pub job_id: Uuid,
    /// Import batch ID (links to database)
    pub import_batch_id: Uuid,
    /// Organization ID
    pub organization_id: Uuid,
    /// Queue ID for progress tracking (PHASE 5)
    pub queue_id: Uuid,
    /// File path to process
    pub file_path: String,
    /// File format
    pub file_format: FileFormat,
    /// File size in bytes
    pub file_size: u64,
    /// SHA-256 hash of file
    pub file_hash: String,
    /// Current processing status
    pub status: ProcessingStatus,
    /// Started processing at
    pub started_at: Option<DateTime<Utc>>,
    /// Completed processing at
    pub completed_at: Option<DateTime<Utc>>,
}

impl IngestionJob {
    pub fn new(
        import_batch_id: Uuid,
        organization_id: Uuid,
        file_path: String,
        file_format: FileFormat,
        file_size: u64,
        file_hash: String,
    ) -> Self {
        Self {
            job_id: Uuid::new_v4(),
            import_batch_id,
            organization_id,
            queue_id: Uuid::new_v4(), // PHASE 5: Generate queue ID for progress tracking
            file_path,
            file_format,
            file_size,
            file_hash,
            status: ProcessingStatus::Queued,
            started_at: None,
            completed_at: None,
        }
    }

    pub fn start(&mut self) {
        self.status = ProcessingStatus::Processing;
        self.started_at = Some(Utc::now());
    }

    pub fn complete(&mut self, status: ProcessingStatus) {
        self.status = status;
        self.completed_at = Some(Utc::now());
    }

    pub fn duration_ms(&self) -> Option<i64> {
        if let (Some(started), Some(completed)) = (self.started_at, self.completed_at) {
            Some((completed - started).num_milliseconds())
        } else {
            None
        }
    }
}

/// Processing statistics for a job
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Total records in file
    pub total_records: usize,
    /// Records successfully parsed
    pub parsed_records: usize,
    /// Records with validation errors
    pub validation_errors: usize,
    /// Records with validation warnings
    pub validation_warnings: usize,
    /// Records that are duplicates
    pub duplicate_records: usize,
    /// Records successfully inserted
    pub inserted_records: usize,
    /// Total flags created
    pub total_flags: usize,
    /// Flags by severity (High, Medium, Low)
    pub flags_by_severity: FlagSeverityCount,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlagSeverityCount {
    pub high: usize,
    pub medium: usize,
    pub low: usize,
}

impl ProcessingStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_records == 0 {
            return 0.0;
        }
        (self.inserted_records as f64 / self.total_records as f64) * 100.0
    }

    pub fn error_rate(&self) -> f64 {
        if self.total_records == 0 {
            return 0.0;
        }
        (self.validation_errors as f64 / self.total_records as f64) * 100.0
    }

    pub fn duplicate_rate(&self) -> f64 {
        if self.total_records == 0 {
            return 0.0;
        }
        (self.duplicate_records as f64 / self.total_records as f64) * 100.0
    }
}

/// Result of processing a single claim/encounter
#[derive(Debug, Clone)]
pub struct ClaimProcessingResult {
    /// Patient control number (unique identifier from source)
    pub patient_control_number: String,
    /// Encounter ID (if successfully inserted)
    pub encounter_id: Option<Uuid>,
    /// Success or failure
    pub success: bool,
    /// Error messages
    pub errors: Vec<String>,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Number of service lines
    pub service_line_count: usize,
    /// Number of flags created
    pub flag_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_format_from_extension() {
        assert_eq!(FileFormat::from_extension("edi"), Some(FileFormat::Edi837p));
        assert_eq!(FileFormat::from_extension("837"), Some(FileFormat::Edi837p));
        assert_eq!(FileFormat::from_extension("csv"), Some(FileFormat::Csv));
        assert_eq!(FileFormat::from_extension("CSV"), Some(FileFormat::Csv));
        assert_eq!(FileFormat::from_extension("unknown"), None);
    }

    #[test]
    fn test_ingestion_job_lifecycle() {
        let mut job = IngestionJob::new(
            Uuid::new_v4(),
            Uuid::new_v4(),
            "/path/to/file.edi".to_string(),
            FileFormat::Edi837p,
            1024,
            "abc123".to_string(),
        );

        assert_eq!(job.status, ProcessingStatus::Queued);
        assert!(job.started_at.is_none());
        assert!(job.completed_at.is_none());

        job.start();
        assert_eq!(job.status, ProcessingStatus::Processing);
        assert!(job.started_at.is_some());

        // Simulate processing delay
        std::thread::sleep(std::time::Duration::from_millis(10));

        job.complete(ProcessingStatus::Completed);
        assert_eq!(job.status, ProcessingStatus::Completed);
        assert!(job.completed_at.is_some());
        assert!(job.duration_ms().is_some());
        assert!(job.duration_ms().unwrap() >= 10);
    }

    #[test]
    fn test_processing_stats() {
        let stats = ProcessingStats {
            total_records: 100,
            parsed_records: 95,
            validation_errors: 5,
            validation_warnings: 10,
            duplicate_records: 3,
            inserted_records: 92,
            total_flags: 15,
            flags_by_severity: FlagSeverityCount {
                high: 5,
                medium: 7,
                low: 3,
            },
        };

        assert_eq!(stats.success_rate(), 92.0);
        assert_eq!(stats.error_rate(), 5.0);
        assert_eq!(stats.duplicate_rate(), 3.0);
    }
}
