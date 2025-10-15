// RVU types for Medicare reimbursement calculation

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// RVU data for a HCPCS/CPT code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RvuData {
    /// HCPCS/CPT code
    pub hcpcs_code: String,
    /// Calendar year
    pub year: i32,
    /// Work RVU - physician work
    pub work_rvu: Decimal,
    /// Practice Expense (PE) RVU - facility
    pub pe_rvu_facility: Decimal,
    /// Practice Expense (PE) RVU - non-facility
    pub pe_rvu_non_facility: Decimal,
    /// Malpractice (MP) RVU
    pub mp_rvu: Decimal,
    /// Total RVU for facility
    pub total_rvu_facility: Decimal,
    /// Total RVU for non-facility
    pub total_rvu_non_facility: Decimal,
    /// Global period (000, 010, 090, XXX, YYY, ZZZ, MMM)
    pub global_period: Option<String>,
    /// Professional component indicator
    pub pc_tc_indicator: Option<String>,
}

impl RvuData {
    /// Get total RVU based on place of service
    pub fn total_rvu(&self, is_facility: bool) -> Decimal {
        if is_facility {
            self.total_rvu_facility
        } else {
            self.total_rvu_non_facility
        }
    }

    /// Get PE RVU based on place of service
    pub fn pe_rvu(&self, is_facility: bool) -> Decimal {
        if is_facility {
            self.pe_rvu_facility
        } else {
            self.pe_rvu_non_facility
        }
    }
}

/// Geographic Practice Cost Index (GPCI) data for a locality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpciData {
    /// Locality code (e.g., "01", "02", "99")
    pub locality_code: String,
    /// Locality name
    pub locality_name: String,
    /// Calendar year
    pub year: i32,
    /// Work GPCI
    pub work_gpci: Decimal,
    /// Practice Expense GPCI
    pub pe_gpci: Decimal,
    /// Malpractice GPCI
    pub mp_gpci: Decimal,
}

/// Medicare Physician Fee Schedule conversion factor
#[derive(Debug, Clone, Copy)]
pub struct ConversionFactor {
    /// Calendar year
    pub year: i32,
    /// Conversion factor dollar amount
    pub factor: Decimal,
}

impl ConversionFactor {
    /// 2024 conversion factor - $33.2875
    pub fn cf_2024() -> ConversionFactor {
        ConversionFactor {
            year: 2024,
            factor: Decimal::new(332875, 4), // 33.2875
        }
    }

    /// 2023 conversion factor - $33.8496
    pub fn cf_2023() -> ConversionFactor {
        ConversionFactor {
            year: 2023,
            factor: Decimal::new(338496, 4), // 33.8496
        }
    }

    /// 2022 conversion factor - $34.2947
    pub fn cf_2022() -> ConversionFactor {
        ConversionFactor {
            year: 2022,
            factor: Decimal::new(342947, 4), // 34.2947
        }
    }

    /// Get conversion factor for a specific year
    pub fn for_year(year: i32) -> Option<ConversionFactor> {
        match year {
            2024 => Some(Self::cf_2024()),
            2023 => Some(Self::cf_2023()),
            2022 => Some(Self::cf_2022()),
            _ => None,
        }
    }
}

/// Modifier adjustment percentages
#[derive(Debug, Clone)]
pub struct ModifierAdjustment {
    pub modifier: String,
    pub adjustment_percent: Decimal,
    pub description: String,
}

impl ModifierAdjustment {
    /// Get adjustment for a specific modifier
    pub fn for_modifier(modifier: &str) -> Option<ModifierAdjustment> {
        match modifier {
            // Bilateral procedures
            "50" => Some(ModifierAdjustment {
                modifier: "50".to_string(),
                adjustment_percent: Decimal::new(150, 2), // 150% (1.50)
                description: "Bilateral procedure".to_string(),
            }),
            // Multiple procedures - second procedure
            "51" => Some(ModifierAdjustment {
                modifier: "51".to_string(),
                adjustment_percent: Decimal::new(50, 2), // 50% (0.50)
                description: "Multiple procedures - reduced by 50%".to_string(),
            }),
            // Reduced services
            "52" => Some(ModifierAdjustment {
                modifier: "52".to_string(),
                adjustment_percent: Decimal::new(50, 2), // 50% (0.50)
                description: "Reduced services".to_string(),
            }),
            // Discontinued procedure
            "53" => Some(ModifierAdjustment {
                modifier: "53".to_string(),
                adjustment_percent: Decimal::new(50, 2), // 50% (0.50)
                description: "Discontinued procedure".to_string(),
            }),
            // Assistant surgeon
            "80" => Some(ModifierAdjustment {
                modifier: "80".to_string(),
                adjustment_percent: Decimal::new(16, 2), // 16% (0.16)
                description: "Assistant surgeon".to_string(),
            }),
            // Minimum assistant surgeon
            "81" => Some(ModifierAdjustment {
                modifier: "81".to_string(),
                adjustment_percent: Decimal::new(16, 2), // 16% (0.16)
                description: "Minimum assistant surgeon".to_string(),
            }),
            // Assistant surgeon (when qualified resident not available)
            "82" => Some(ModifierAdjustment {
                modifier: "82".to_string(),
                adjustment_percent: Decimal::new(16, 2), // 16% (0.16)
                description: "Assistant surgeon (no resident)".to_string(),
            }),
            // Professional component
            "26" => Some(ModifierAdjustment {
                modifier: "26".to_string(),
                adjustment_percent: Decimal::new(100, 2), // 100% (1.00) - uses work RVU only
                description: "Professional component only".to_string(),
            }),
            // Technical component
            "TC" => Some(ModifierAdjustment {
                modifier: "TC".to_string(),
                adjustment_percent: Decimal::new(100, 2), // 100% (1.00) - uses PE/MP RVU only
                description: "Technical component only".to_string(),
            }),
            // Co-surgeon
            "62" => Some(ModifierAdjustment {
                modifier: "62".to_string(),
                adjustment_percent: Decimal::new(62_5, 3), // 62.5% (0.625)
                description: "Co-surgeon".to_string(),
            }),
            // Team surgery
            "66" => Some(ModifierAdjustment {
                modifier: "66".to_string(),
                adjustment_percent: Decimal::new(100, 2), // 100% (1.00) - by report
                description: "Team surgery - by report".to_string(),
            }),
            _ => None,
        }
    }

    /// Check if modifier affects payment amount
    pub fn affects_payment(modifier: &str) -> bool {
        Self::for_modifier(modifier).is_some()
    }
}

/// Place of Service categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaceOfService {
    /// Facility (hospital inpatient, outpatient, ASC, SNF, etc.)
    Facility,
    /// Non-facility (office, clinic, home, etc.)
    NonFacility,
}

impl PlaceOfService {
    /// Determine if a POS code is facility or non-facility
    pub fn from_code(code: &str) -> Option<PlaceOfService> {
        match code {
            // Facility codes
            "21" | // Inpatient hospital
            "22" | // Outpatient hospital
            "23" | // Emergency room - hospital
            "24" | // Ambulatory surgical center
            "31" | // Skilled nursing facility
            "32" | // Nursing facility
            "34" | // Hospice
            "51" | // Inpatient psychiatric facility
            "52" | // Psychiatric facility - partial
            "53" | // Community mental health center
            "56" | // Psychiatric residential treatment center
            "61" => Some(PlaceOfService::Facility), // Comprehensive inpatient rehab facility

            // Non-facility codes
            "01" | // Pharmacy
            "02" | // Telehealth
            "03" | // School
            "04" | // Homeless shelter
            "11" | // Office
            "12" | // Home
            "13" | // Assisted living facility
            "14" | // Group home
            "15" | // Mobile unit
            "16" | // Temporary lodging
            "17" | // Walk-in retail health clinic
            "19" | // Off campus-outpatient hospital
            "20" | // Urgent care facility
            "25" | // Birthing center
            "26" | // Military treatment facility
            "33" | // Custodial care facility
            "41" | // Ambulance - land
            "42" | // Ambulance - air or water
            "49" | // Independent clinic
            "50" | // Federally qualified health center
            "54" | // Intermediate care facility/mentally retarded
            "55" | // Residential substance abuse treatment facility
            "57" | // Non-residential substance abuse treatment facility
            "60" | // Mass immunization center
            "62" | // Comprehensive outpatient rehab facility
            "65" | // End-stage renal disease treatment facility
            "71" | // Public health clinic
            "72" | // Rural health clinic
            "81" | // Independent laboratory
            "99" => Some(PlaceOfService::NonFacility), // Other place of service

            _ => None,
        }
    }

    /// Check if code is facility
    pub fn is_facility(code: &str) -> bool {
        matches!(Self::from_code(code), Some(PlaceOfService::Facility))
    }

    /// Check if code is non-facility
    pub fn is_non_facility(code: &str) -> bool {
        matches!(Self::from_code(code), Some(PlaceOfService::NonFacility))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rvu_data_total() {
        let rvu = RvuData {
            hcpcs_code: "99213".to_string(),
            year: 2024,
            work_rvu: Decimal::new(130, 2), // 1.30
            pe_rvu_facility: Decimal::new(60, 2), // 0.60
            pe_rvu_non_facility: Decimal::new(120, 2), // 1.20
            mp_rvu: Decimal::new(8, 2), // 0.08
            total_rvu_facility: Decimal::new(198, 2), // 1.98
            total_rvu_non_facility: Decimal::new(258, 2), // 2.58
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        };

        assert_eq!(rvu.total_rvu(true), Decimal::new(198, 2));
        assert_eq!(rvu.total_rvu(false), Decimal::new(258, 2));
        assert_eq!(rvu.pe_rvu(true), Decimal::new(60, 2));
        assert_eq!(rvu.pe_rvu(false), Decimal::new(120, 2));
    }

    #[test]
    fn test_conversion_factor_lookup() {
        let cf_2024 = ConversionFactor::for_year(2024);
        assert!(cf_2024.is_some());
        assert_eq!(cf_2024.unwrap().factor, Decimal::new(332875, 4));

        let cf_2023 = ConversionFactor::for_year(2023);
        assert!(cf_2023.is_some());
        assert_eq!(cf_2023.unwrap().factor, Decimal::new(338496, 4));

        let cf_2022 = ConversionFactor::for_year(2022);
        assert!(cf_2022.is_some());
        assert_eq!(cf_2022.unwrap().factor, Decimal::new(342947, 4));

        let cf_2020 = ConversionFactor::for_year(2020);
        assert!(cf_2020.is_none());
    }

    #[test]
    fn test_modifier_adjustments() {
        let mod_50 = ModifierAdjustment::for_modifier("50");
        assert!(mod_50.is_some());
        assert_eq!(mod_50.unwrap().adjustment_percent, Decimal::new(150, 2));

        let mod_51 = ModifierAdjustment::for_modifier("51");
        assert!(mod_51.is_some());
        assert_eq!(mod_51.unwrap().adjustment_percent, Decimal::new(50, 2));

        let mod_invalid = ModifierAdjustment::for_modifier("99");
        assert!(mod_invalid.is_none());
    }

    #[test]
    fn test_modifier_affects_payment() {
        assert!(ModifierAdjustment::affects_payment("50"));
        assert!(ModifierAdjustment::affects_payment("51"));
        assert!(ModifierAdjustment::affects_payment("26"));
        assert!(!ModifierAdjustment::affects_payment("25"));
        assert!(!ModifierAdjustment::affects_payment("99"));
    }

    #[test]
    fn test_place_of_service() {
        assert_eq!(PlaceOfService::from_code("21"), Some(PlaceOfService::Facility));
        assert_eq!(PlaceOfService::from_code("22"), Some(PlaceOfService::Facility));
        assert_eq!(PlaceOfService::from_code("11"), Some(PlaceOfService::NonFacility));
        assert_eq!(PlaceOfService::from_code("12"), Some(PlaceOfService::NonFacility));
        assert_eq!(PlaceOfService::from_code("999"), None);

        assert!(PlaceOfService::is_facility("21"));
        assert!(PlaceOfService::is_non_facility("11"));
        assert!(!PlaceOfService::is_facility("11"));
        assert!(!PlaceOfService::is_non_facility("21"));
    }
}
