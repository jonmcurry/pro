// Flag types and categories for healthcare claim auditing

use serde::{Deserialize, Serialize};
use std::fmt;


/// Flag categories - 11 main categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlagCategory {
    /// COD: Coding Issues
    Coding,
    /// DOC: Documentation Issues
    Documentation,
    /// EMO: E/M Over-coded
    EMOvercoded,
    /// EMU: E/M Under-coded
    EMUndercoded,
    /// EMI: E/M Incorrect Category
    EMIncorrectCategory,
    /// EMT: E/M Time Not Documented
    EMTimeNotDocumented,
    /// MOD: Modifier Issues
    Modifier,
    /// OTH: Other Issues
    Other,
    /// QTY: Quantity Issues
    Quantity,
    /// SUP: Supervision Requirements
    Supervision,
    /// DX: Diagnosis Issues
    Diagnosis,
}

impl FlagCategory {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Coding => "COD",
            Self::Documentation => "DOC",
            Self::EMOvercoded => "EMO",
            Self::EMUndercoded => "EMU",
            Self::EMIncorrectCategory => "EMI",
            Self::EMTimeNotDocumented => "EMT",
            Self::Modifier => "MOD",
            Self::Other => "OTH",
            Self::Quantity => "QTY",
            Self::Supervision => "SUP",
            Self::Diagnosis => "DX",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Coding => "Coding Issues",
            Self::Documentation => "Documentation Issues",
            Self::EMOvercoded => "E/M Over-coded",
            Self::EMUndercoded => "E/M Under-coded",
            Self::EMIncorrectCategory => "E/M Incorrect Category",
            Self::EMTimeNotDocumented => "E/M Time Not Documented",
            Self::Modifier => "Modifier Issues",
            Self::Other => "Other Issues",
            Self::Quantity => "Quantity Issues",
            Self::Supervision => "Supervision Requirements",
            Self::Diagnosis => "Diagnosis Issues",
        }
    }
}

impl fmt::Display for FlagCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.name())
    }
}

/// Flag issue types - 27 specific types across all categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlagIssueType {
    // COD: Coding Issues (4 types)
    /// Incorrect procedure code selected
    CodIncorrectProcedureCode,
    /// Procedure code not supported by diagnosis
    CodProcedureNotSupportedByDiagnosis,
    /// Unbundling detected (separate billing for bundled procedures)
    CodUnbundling,
    /// Upcoding detected (billing higher level service than documented)
    CodUpcoding,

    // DOC: Documentation Issues (2 types)
    /// Insufficient documentation to support level of service
    DocInsufficientDocumentation,
    /// Missing required documentation elements
    DocMissingRequiredElements,

    // EMO: E/M Over-coded (2 types)
    /// E/M level higher than MDM supports
    EMOLevelHigherThanMDM,
    /// E/M level higher than history/exam supports
    EMOLevelHigherThanHistoryExam,

    // EMU: E/M Under-coded (2 types)
    /// E/M level lower than MDM supports
    EMULevelLowerThanMDM,
    /// E/M level lower than time spent supports
    EMULevelLowerThanTime,

    // EMI: E/M Incorrect Category (1 type)
    /// Wrong E/M category selected (e.g., new vs established)
    EMIWrongCategory,

    // EMT: E/M Time Not Documented (1 type)
    /// Time-based E/M code without documented time
    EMTTimeNotDocumented,

    // MOD: Modifier Issues (3 types)
    /// Missing required modifier
    ModMissingRequired,
    /// Incorrect modifier applied
    ModIncorrect,
    /// Conflicting modifiers on same service
    ModConflicting,

    // OTH: Other Issues (3 types)
    /// Medical necessity not established
    OthMedicalNecessityNotEstablished,
    /// Service rendered by wrong provider type
    OthWrongProviderType,
    /// Duplicate service billing
    OthDuplicateService,

    // QTY: Quantity Issues (2 types)
    /// Units billed exceed reasonable maximum
    QtyUnitsExceedMaximum,
    /// Units inconsistent with documentation
    QtyUnitsInconsistent,

    // SUP: Supervision Requirements (3 types)
    /// Service requires supervision not documented
    SupSupervisionNotDocumented,
    /// Inappropriate supervision level
    SupInappropriateLevel,
    /// Teaching physician requirements not met
    SupTeachingPhysicianNotMet,

    // DX: Diagnosis Issues (4 types)
    /// Primary diagnosis does not support service
    DxPrimaryDoesNotSupport,
    /// Missing specificity in diagnosis coding
    DxMissingSpecificity,
    /// Diagnosis code sequencing error
    DxSequencingError,
    /// Unspecified diagnosis code used when specific available
    DxUnspecifiedWhenSpecificAvailable,
}

impl FlagIssueType {
    pub fn category(&self) -> FlagCategory {
        match self {
            Self::CodIncorrectProcedureCode
            | Self::CodProcedureNotSupportedByDiagnosis
            | Self::CodUnbundling
            | Self::CodUpcoding => FlagCategory::Coding,

            Self::DocInsufficientDocumentation | Self::DocMissingRequiredElements => {
                FlagCategory::Documentation
            }

            Self::EMOLevelHigherThanMDM | Self::EMOLevelHigherThanHistoryExam => {
                FlagCategory::EMOvercoded
            }

            Self::EMULevelLowerThanMDM | Self::EMULevelLowerThanTime => {
                FlagCategory::EMUndercoded
            }

            Self::EMIWrongCategory => FlagCategory::EMIncorrectCategory,

            Self::EMTTimeNotDocumented => FlagCategory::EMTimeNotDocumented,

            Self::ModMissingRequired | Self::ModIncorrect | Self::ModConflicting => {
                FlagCategory::Modifier
            }

            Self::OthMedicalNecessityNotEstablished
            | Self::OthWrongProviderType
            | Self::OthDuplicateService => FlagCategory::Other,

            Self::QtyUnitsExceedMaximum | Self::QtyUnitsInconsistent => FlagCategory::Quantity,

            Self::SupSupervisionNotDocumented
            | Self::SupInappropriateLevel
            | Self::SupTeachingPhysicianNotMet => FlagCategory::Supervision,

            Self::DxPrimaryDoesNotSupport
            | Self::DxMissingSpecificity
            | Self::DxSequencingError
            | Self::DxUnspecifiedWhenSpecificAvailable => FlagCategory::Diagnosis,
        }
    }

    pub fn code(&self) -> &'static str {
        match self {
            Self::CodIncorrectProcedureCode => "COD-001",
            Self::CodProcedureNotSupportedByDiagnosis => "COD-002",
            Self::CodUnbundling => "COD-003",
            Self::CodUpcoding => "COD-004",

            Self::DocInsufficientDocumentation => "DOC-001",
            Self::DocMissingRequiredElements => "DOC-002",

            Self::EMOLevelHigherThanMDM => "EMO-001",
            Self::EMOLevelHigherThanHistoryExam => "EMO-002",

            Self::EMULevelLowerThanMDM => "EMU-001",
            Self::EMULevelLowerThanTime => "EMU-002",

            Self::EMIWrongCategory => "EMI-001",

            Self::EMTTimeNotDocumented => "EMT-001",

            Self::ModMissingRequired => "MOD-001",
            Self::ModIncorrect => "MOD-002",
            Self::ModConflicting => "MOD-003",

            Self::OthMedicalNecessityNotEstablished => "OTH-001",
            Self::OthWrongProviderType => "OTH-002",
            Self::OthDuplicateService => "OTH-003",

            Self::QtyUnitsExceedMaximum => "QTY-001",
            Self::QtyUnitsInconsistent => "QTY-002",

            Self::SupSupervisionNotDocumented => "SUP-001",
            Self::SupInappropriateLevel => "SUP-002",
            Self::SupTeachingPhysicianNotMet => "SUP-003",

            Self::DxPrimaryDoesNotSupport => "DX-001",
            Self::DxMissingSpecificity => "DX-002",
            Self::DxSequencingError => "DX-003",
            Self::DxUnspecifiedWhenSpecificAvailable => "DX-004",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::CodIncorrectProcedureCode => "Incorrect Procedure Code",
            Self::CodProcedureNotSupportedByDiagnosis => "Procedure Not Supported by Diagnosis",
            Self::CodUnbundling => "Unbundling Detected",
            Self::CodUpcoding => "Upcoding Detected",

            Self::DocInsufficientDocumentation => "Insufficient Documentation",
            Self::DocMissingRequiredElements => "Missing Required Elements",

            Self::EMOLevelHigherThanMDM => "E/M Level Higher Than MDM",
            Self::EMOLevelHigherThanHistoryExam => "E/M Level Higher Than History/Exam",

            Self::EMULevelLowerThanMDM => "E/M Level Lower Than MDM",
            Self::EMULevelLowerThanTime => "E/M Level Lower Than Time",

            Self::EMIWrongCategory => "Wrong E/M Category",

            Self::EMTTimeNotDocumented => "Time Not Documented",

            Self::ModMissingRequired => "Missing Required Modifier",
            Self::ModIncorrect => "Incorrect Modifier",
            Self::ModConflicting => "Conflicting Modifiers",

            Self::OthMedicalNecessityNotEstablished => "Medical Necessity Not Established",
            Self::OthWrongProviderType => "Wrong Provider Type",
            Self::OthDuplicateService => "Duplicate Service",

            Self::QtyUnitsExceedMaximum => "Units Exceed Maximum",
            Self::QtyUnitsInconsistent => "Units Inconsistent",

            Self::SupSupervisionNotDocumented => "Supervision Not Documented",
            Self::SupInappropriateLevel => "Inappropriate Supervision Level",
            Self::SupTeachingPhysicianNotMet => "Teaching Physician Requirements Not Met",

            Self::DxPrimaryDoesNotSupport => "Primary Diagnosis Does Not Support Service",
            Self::DxMissingSpecificity => "Missing Diagnosis Specificity",
            Self::DxSequencingError => "Diagnosis Sequencing Error",
            Self::DxUnspecifiedWhenSpecificAvailable => "Unspecified Code When Specific Available",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::CodIncorrectProcedureCode => {
                "The procedure code selected does not accurately represent the service documented"
            }
            Self::CodProcedureNotSupportedByDiagnosis => {
                "The documented diagnoses do not support the medical necessity of the procedure"
            }
            Self::CodUnbundling => {
                "Separate billing for procedures that should be billed as a single comprehensive code"
            }
            Self::CodUpcoding => {
                "Billing a higher level of service than what was documented or provided"
            }

            Self::DocInsufficientDocumentation => {
                "Documentation does not contain sufficient detail to support the level of service billed"
            }
            Self::DocMissingRequiredElements => {
                "Required documentation elements are missing (e.g., chief complaint, HPI, exam)"
            }

            Self::EMOLevelHigherThanMDM => {
                "E/M level billed is higher than supported by medical decision making complexity"
            }
            Self::EMOLevelHigherThanHistoryExam => {
                "E/M level billed is higher than supported by history and examination"
            }

            Self::EMULevelLowerThanMDM => {
                "E/M level billed is lower than justified by medical decision making complexity"
            }
            Self::EMULevelLowerThanTime => {
                "E/M level billed is lower than justified by time spent on the encounter"
            }

            Self::EMIWrongCategory => {
                "Incorrect E/M category selected (e.g., new patient code for established patient)"
            }

            Self::EMTTimeNotDocumented => {
                "Time-based E/M code used without documented time spent on encounter"
            }

            Self::ModMissingRequired => {
                "Required modifier is missing from the service line"
            }
            Self::ModIncorrect => {
                "Incorrect modifier applied to the service"
            }
            Self::ModConflicting => {
                "Conflicting modifiers present on the same service line"
            }

            Self::OthMedicalNecessityNotEstablished => {
                "Documentation does not establish medical necessity for the service"
            }
            Self::OthWrongProviderType => {
                "Service rendered by provider type not authorized for this service"
            }
            Self::OthDuplicateService => {
                "Service appears to be billed multiple times for the same date"
            }

            Self::QtyUnitsExceedMaximum => {
                "Units billed exceed reasonable maximum for this service and timeframe"
            }
            Self::QtyUnitsInconsistent => {
                "Units billed are inconsistent with documentation"
            }

            Self::SupSupervisionNotDocumented => {
                "Service requires supervision but supervision is not documented"
            }
            Self::SupInappropriateLevel => {
                "Level of supervision documented is inappropriate for this service"
            }
            Self::SupTeachingPhysicianNotMet => {
                "Teaching physician requirements for resident services not met"
            }

            Self::DxPrimaryDoesNotSupport => {
                "Primary diagnosis does not support the service rendered"
            }
            Self::DxMissingSpecificity => {
                "Diagnosis lacks required specificity (e.g., laterality, encounter type)"
            }
            Self::DxSequencingError => {
                "Diagnosis codes not sequenced correctly (e.g., primary diagnosis is secondary condition)"
            }
            Self::DxUnspecifiedWhenSpecificAvailable => {
                "Unspecified diagnosis code used when a more specific code is available"
            }
        }
    }

    /// Get default severity for this flag type
    pub fn default_severity(&self) -> FlagSeverity {
        match self {
            // High severity - potential fraud or major compliance issues
            Self::CodUpcoding | Self::CodUnbundling | Self::OthDuplicateService => {
                FlagSeverity::High
            }

            // Medium severity - significant coding/documentation issues
            Self::CodIncorrectProcedureCode
            | Self::CodProcedureNotSupportedByDiagnosis
            | Self::DocInsufficientDocumentation
            | Self::DocMissingRequiredElements
            | Self::EMOLevelHigherThanMDM
            | Self::EMOLevelHigherThanHistoryExam
            | Self::EMIWrongCategory
            | Self::EMTTimeNotDocumented
            | Self::ModMissingRequired
            | Self::ModIncorrect
            | Self::OthMedicalNecessityNotEstablished
            | Self::OthWrongProviderType
            | Self::QtyUnitsExceedMaximum
            | Self::SupSupervisionNotDocumented
            | Self::SupInappropriateLevel
            | Self::SupTeachingPhysicianNotMet
            | Self::DxPrimaryDoesNotSupport
            | Self::DxSequencingError => FlagSeverity::Medium,

            // Low severity - educational/optimization opportunities
            Self::EMULevelLowerThanMDM
            | Self::EMULevelLowerThanTime
            | Self::ModConflicting
            | Self::QtyUnitsInconsistent
            | Self::DxMissingSpecificity
            | Self::DxUnspecifiedWhenSpecificAvailable => FlagSeverity::Low,
        }
    }

    /// Get all flag types
    pub fn all() -> Vec<Self> {
        vec![
            // Coding (4)
            Self::CodIncorrectProcedureCode,
            Self::CodProcedureNotSupportedByDiagnosis,
            Self::CodUnbundling,
            Self::CodUpcoding,
            // Documentation (2)
            Self::DocInsufficientDocumentation,
            Self::DocMissingRequiredElements,
            // E/M Overcoded (2)
            Self::EMOLevelHigherThanMDM,
            Self::EMOLevelHigherThanHistoryExam,
            // E/M Undercoded (2)
            Self::EMULevelLowerThanMDM,
            Self::EMULevelLowerThanTime,
            // E/M Incorrect Category (1)
            Self::EMIWrongCategory,
            // E/M Time Not Documented (1)
            Self::EMTTimeNotDocumented,
            // Modifier (3)
            Self::ModMissingRequired,
            Self::ModIncorrect,
            Self::ModConflicting,
            // Other (3)
            Self::OthMedicalNecessityNotEstablished,
            Self::OthWrongProviderType,
            Self::OthDuplicateService,
            // Quantity (2)
            Self::QtyUnitsExceedMaximum,
            Self::QtyUnitsInconsistent,
            // Supervision (3)
            Self::SupSupervisionNotDocumented,
            Self::SupInappropriateLevel,
            Self::SupTeachingPhysicianNotMet,
            // Diagnosis (4)
            Self::DxPrimaryDoesNotSupport,
            Self::DxMissingSpecificity,
            Self::DxSequencingError,
            Self::DxUnspecifiedWhenSpecificAvailable,
        ]
    }

    /// Get flag types by category
    pub fn by_category(category: FlagCategory) -> Vec<Self> {
        Self::all()
            .into_iter()
            .filter(|t| t.category() == category)
            .collect()
    }
}

impl fmt::Display for FlagIssueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.name())
    }
}

/// Flag severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum FlagSeverity {
    Low,
    Medium,
    High,
}

impl FlagSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "LOW",
            Self::Medium => "MEDIUM",
            Self::High => "HIGH",
        }
    }
}

impl fmt::Display for FlagSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Context information for flag creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlagContext {
    pub encounter_id: Option<i64>,
    pub service_line_id: Option<i64>,
    pub organization_id: i64,
    pub facility_id: Option<i64>,
    pub provider_id: Option<i64>,
    pub coder_id: Option<i64>,
    pub procedure_code: Option<String>,
    pub diagnosis_codes: Vec<String>,
    pub modifiers: Vec<String>,
    pub units: Option<rust_decimal::Decimal>,
    pub charge_amount: Option<rust_decimal::Decimal>,
}

impl FlagContext {
    pub fn new(organization_id: i64) -> Self {
        Self {
            encounter_id: None,
            service_line_id: None,
            organization_id,
            facility_id: None,
            provider_id: None,
            coder_id: None,
            procedure_code: None,
            diagnosis_codes: Vec::new(),
            modifiers: Vec::new(),
            units: None,
            charge_amount: None,
        }
    }

    pub fn with_encounter(mut self, encounter_id: i64) -> Self {
        self.encounter_id = Some(encounter_id);
        self
    }

    pub fn with_service_line(mut self, service_line_id: i64) -> Self {
        self.service_line_id = Some(service_line_id);
        self
    }

    pub fn with_provider(mut self, provider_id: i64) -> Self {
        self.provider_id = Some(provider_id);
        self
    }

    pub fn with_procedure_code(mut self, procedure_code: String) -> Self {
        self.procedure_code = Some(procedure_code);
        self
    }

    pub fn with_diagnosis_codes(mut self, diagnosis_codes: Vec<String>) -> Self {
        self.diagnosis_codes = diagnosis_codes;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flag_category_count() {
        // Verify we have 11 categories
        let categories = vec![
            FlagCategory::Coding,
            FlagCategory::Documentation,
            FlagCategory::EMOvercoded,
            FlagCategory::EMUndercoded,
            FlagCategory::EMIncorrectCategory,
            FlagCategory::EMTimeNotDocumented,
            FlagCategory::Modifier,
            FlagCategory::Other,
            FlagCategory::Quantity,
            FlagCategory::Supervision,
            FlagCategory::Diagnosis,
        ];
        assert_eq!(categories.len(), 11);
    }

    #[test]
    fn test_flag_issue_type_count() {
        // Verify we have 27 flag types total
        let all_types = FlagIssueType::all();
        assert_eq!(all_types.len(), 27);
    }

    #[test]
    fn test_flag_types_by_category() {
        // COD: 4 types
        assert_eq!(FlagIssueType::by_category(FlagCategory::Coding).len(), 4);

        // DOC: 2 types
        assert_eq!(
            FlagIssueType::by_category(FlagCategory::Documentation).len(),
            2
        );

        // EMO: 2 types
        assert_eq!(
            FlagIssueType::by_category(FlagCategory::EMOvercoded).len(),
            2
        );

        // EMU: 2 types
        assert_eq!(
            FlagIssueType::by_category(FlagCategory::EMUndercoded).len(),
            2
        );

        // EMI: 1 type
        assert_eq!(
            FlagIssueType::by_category(FlagCategory::EMIncorrectCategory).len(),
            1
        );

        // EMT: 1 type
        assert_eq!(
            FlagIssueType::by_category(FlagCategory::EMTimeNotDocumented).len(),
            1
        );

        // MOD: 3 types
        assert_eq!(FlagIssueType::by_category(FlagCategory::Modifier).len(), 3);

        // OTH: 3 types
        assert_eq!(FlagIssueType::by_category(FlagCategory::Other).len(), 3);

        // QTY: 2 types
        assert_eq!(FlagIssueType::by_category(FlagCategory::Quantity).len(), 2);

        // SUP: 3 types
        assert_eq!(
            FlagIssueType::by_category(FlagCategory::Supervision).len(),
            3
        );

        // DX: 4 types
        assert_eq!(FlagIssueType::by_category(FlagCategory::Diagnosis).len(), 4);
    }

    #[test]
    fn test_flag_codes_unique() {
        let all_types = FlagIssueType::all();
        let codes: Vec<&str> = all_types.iter().map(|t| t.code()).collect();
        let unique_codes: std::collections::HashSet<&str> = codes.iter().cloned().collect();

        assert_eq!(codes.len(), unique_codes.len(), "All flag codes should be unique");
    }

    #[test]
    fn test_flag_category_mapping() {
        // Test that each flag type returns correct category
        assert_eq!(
            FlagIssueType::CodIncorrectProcedureCode.category(),
            FlagCategory::Coding
        );
        assert_eq!(
            FlagIssueType::DocInsufficientDocumentation.category(),
            FlagCategory::Documentation
        );
        assert_eq!(
            FlagIssueType::EMOLevelHigherThanMDM.category(),
            FlagCategory::EMOvercoded
        );
        assert_eq!(
            FlagIssueType::ModMissingRequired.category(),
            FlagCategory::Modifier
        );
        assert_eq!(
            FlagIssueType::DxPrimaryDoesNotSupport.category(),
            FlagCategory::Diagnosis
        );
    }

    #[test]
    fn test_flag_severity() {
        // High severity
        assert_eq!(
            FlagIssueType::CodUpcoding.default_severity(),
            FlagSeverity::High
        );
        assert_eq!(
            FlagIssueType::CodUnbundling.default_severity(),
            FlagSeverity::High
        );

        // Medium severity
        assert_eq!(
            FlagIssueType::CodIncorrectProcedureCode.default_severity(),
            FlagSeverity::Medium
        );

        // Low severity
        assert_eq!(
            FlagIssueType::EMULevelLowerThanMDM.default_severity(),
            FlagSeverity::Low
        );
    }

    #[test]
    fn test_flag_context_builder() {
        let org_id = 1i64;
        let encounter_id = 1i64;
        let provider_id = 1i64;

        let context = FlagContext::new(org_id)
            .with_encounter(encounter_id)
            .with_provider(provider_id)
            .with_procedure_code("99213".to_string())
            .with_diagnosis_codes(vec!["E11.9".to_string(), "I10".to_string()]);

        assert_eq!(context.organization_id, org_id);
        assert_eq!(context.encounter_id, Some(encounter_id));
        assert_eq!(context.provider_id, Some(provider_id));
        assert_eq!(context.procedure_code, Some("99213".to_string()));
        assert_eq!(context.diagnosis_codes.len(), 2);
    }
}
