// RVU and GPCI lookup functionality

use crate::types::{GpciData, RvuData};
use pro_common::{Error, Result};
use rust_decimal::Decimal;
use std::collections::HashMap;

/// RVU lookup service
pub struct RvuLookup {
    /// RVU data indexed by (hcpcs_code, year)
    rvu_data: HashMap<(String, i32), RvuData>,
}

impl RvuLookup {
    /// Create a new RVU lookup service
    pub fn new() -> Self {
        Self {
            rvu_data: HashMap::new(),
        }
    }

    /// Create with pre-populated data
    pub fn with_data(data: Vec<RvuData>) -> Self {
        let mut lookup = Self::new();
        for rvu in data {
            lookup.add_rvu_data(rvu);
        }
        lookup
    }

    /// Add RVU data to the lookup
    pub fn add_rvu_data(&mut self, rvu: RvuData) {
        let key = (rvu.hcpcs_code.clone(), rvu.year);
        self.rvu_data.insert(key, rvu);
    }

    /// Lookup RVU data by HCPCS code and year
    pub fn lookup(&self, hcpcs_code: &str, year: i32) -> Result<&RvuData> {
        let key = (hcpcs_code.to_string(), year);
        self.rvu_data
            .get(&key)
            .ok_or_else(|| Error::Validation(format!(
                "RVU data not found for code {} in year {}",
                hcpcs_code, year
            )))
    }

    /// Check if RVU data exists
    pub fn has_data(&self, hcpcs_code: &str, year: i32) -> bool {
        let key = (hcpcs_code.to_string(), year);
        self.rvu_data.contains_key(&key)
    }

    /// Get total count of RVU entries
    pub fn count(&self) -> usize {
        self.rvu_data.len()
    }

    /// Load sample 2024 E/M data for testing/demo
    pub fn load_sample_2024_em_data(&mut self) {
        // Office visits - new patient
        self.add_rvu_data(RvuData {
            hcpcs_code: "99202".to_string(),
            year: 2024,
            work_rvu: Decimal::new(93, 2), // 0.93
            pe_rvu_facility: Decimal::new(45, 2), // 0.45
            pe_rvu_non_facility: Decimal::new(99, 2), // 0.99
            mp_rvu: Decimal::new(5, 2), // 0.05
            total_rvu_facility: Decimal::new(143, 2), // 1.43
            total_rvu_non_facility: Decimal::new(197, 2), // 1.97
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        self.add_rvu_data(RvuData {
            hcpcs_code: "99203".to_string(),
            year: 2024,
            work_rvu: Decimal::new(130, 2), // 1.30
            pe_rvu_facility: Decimal::new(60, 2), // 0.60
            pe_rvu_non_facility: Decimal::new(144, 2), // 1.44
            mp_rvu: Decimal::new(8, 2), // 0.08
            total_rvu_facility: Decimal::new(198, 2), // 1.98
            total_rvu_non_facility: Decimal::new(282, 2), // 2.82
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        self.add_rvu_data(RvuData {
            hcpcs_code: "99204".to_string(),
            year: 2024,
            work_rvu: Decimal::new(192, 2), // 1.92
            pe_rvu_facility: Decimal::new(89, 2), // 0.89
            pe_rvu_non_facility: Decimal::new(208, 2), // 2.08
            mp_rvu: Decimal::new(12, 2), // 0.12
            total_rvu_facility: Decimal::new(293, 2), // 2.93
            total_rvu_non_facility: Decimal::new(412, 2), // 4.12
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        self.add_rvu_data(RvuData {
            hcpcs_code: "99205".to_string(),
            year: 2024,
            work_rvu: Decimal::new(280, 2), // 2.80
            pe_rvu_facility: Decimal::new(128, 2), // 1.28
            pe_rvu_non_facility: Decimal::new(294, 2), // 2.94
            mp_rvu: Decimal::new(17, 2), // 0.17
            total_rvu_facility: Decimal::new(425, 2), // 4.25
            total_rvu_non_facility: Decimal::new(591, 2), // 5.91
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        // Office visits - established patient
        self.add_rvu_data(RvuData {
            hcpcs_code: "99211".to_string(),
            year: 2024,
            work_rvu: Decimal::new(18, 2), // 0.18
            pe_rvu_facility: Decimal::new(14, 2), // 0.14
            pe_rvu_non_facility: Decimal::new(36, 2), // 0.36
            mp_rvu: Decimal::new(1, 2), // 0.01
            total_rvu_facility: Decimal::new(33, 2), // 0.33
            total_rvu_non_facility: Decimal::new(55, 2), // 0.55
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        self.add_rvu_data(RvuData {
            hcpcs_code: "99212".to_string(),
            year: 2024,
            work_rvu: Decimal::new(70, 2), // 0.70
            pe_rvu_facility: Decimal::new(32, 2), // 0.32
            pe_rvu_non_facility: Decimal::new(74, 2), // 0.74
            mp_rvu: Decimal::new(4, 2), // 0.04
            total_rvu_facility: Decimal::new(106, 2), // 1.06
            total_rvu_non_facility: Decimal::new(148, 2), // 1.48
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        self.add_rvu_data(RvuData {
            hcpcs_code: "99213".to_string(),
            year: 2024,
            work_rvu: Decimal::new(110, 2), // 1.10
            pe_rvu_facility: Decimal::new(48, 2), // 0.48
            pe_rvu_non_facility: Decimal::new(113, 2), // 1.13
            mp_rvu: Decimal::new(7, 2), // 0.07
            total_rvu_facility: Decimal::new(165, 2), // 1.65
            total_rvu_non_facility: Decimal::new(230, 2), // 2.30
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        self.add_rvu_data(RvuData {
            hcpcs_code: "99214".to_string(),
            year: 2024,
            work_rvu: Decimal::new(166, 2), // 1.66
            pe_rvu_facility: Decimal::new(72, 2), // 0.72
            pe_rvu_non_facility: Decimal::new(165, 2), // 1.65
            mp_rvu: Decimal::new(10, 2), // 0.10
            total_rvu_facility: Decimal::new(248, 2), // 2.48
            total_rvu_non_facility: Decimal::new(341, 2), // 3.41
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        self.add_rvu_data(RvuData {
            hcpcs_code: "99215".to_string(),
            year: 2024,
            work_rvu: Decimal::new(242, 2), // 2.42
            pe_rvu_facility: Decimal::new(104, 2), // 1.04
            pe_rvu_non_facility: Decimal::new(236, 2), // 2.36
            mp_rvu: Decimal::new(15, 2), // 0.15
            total_rvu_facility: Decimal::new(361, 2), // 3.61
            total_rvu_non_facility: Decimal::new(493, 2), // 4.93
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        // Initial hospital care
        self.add_rvu_data(RvuData {
            hcpcs_code: "99221".to_string(),
            year: 2024,
            work_rvu: Decimal::new(130, 2), // 1.30
            pe_rvu_facility: Decimal::new(72, 2), // 0.72
            pe_rvu_non_facility: Decimal::new(72, 2), // 0.72
            mp_rvu: Decimal::new(15, 2), // 0.15
            total_rvu_facility: Decimal::new(217, 2), // 2.17
            total_rvu_non_facility: Decimal::new(217, 2), // 2.17
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        self.add_rvu_data(RvuData {
            hcpcs_code: "99222".to_string(),
            year: 2024,
            work_rvu: Decimal::new(200, 2), // 2.00
            pe_rvu_facility: Decimal::new(107, 2), // 1.07
            pe_rvu_non_facility: Decimal::new(107, 2), // 1.07
            mp_rvu: Decimal::new(23, 2), // 0.23
            total_rvu_facility: Decimal::new(330, 2), // 3.30
            total_rvu_non_facility: Decimal::new(330, 2), // 3.30
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        self.add_rvu_data(RvuData {
            hcpcs_code: "99223".to_string(),
            year: 2024,
            work_rvu: Decimal::new(307, 2), // 3.07
            pe_rvu_facility: Decimal::new(158, 2), // 1.58
            pe_rvu_non_facility: Decimal::new(158, 2), // 1.58
            mp_rvu: Decimal::new(35, 2), // 0.35
            total_rvu_facility: Decimal::new(500, 2), // 5.00
            total_rvu_non_facility: Decimal::new(500, 2), // 5.00
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });
    }
}

impl Default for RvuLookup {
    fn default() -> Self {
        Self::new()
    }
}

/// GPCI lookup service
pub struct GpciLookup {
    /// GPCI data indexed by (locality_code, year)
    gpci_data: HashMap<(String, i32), GpciData>,
}

impl GpciLookup {
    /// Create a new GPCI lookup service
    pub fn new() -> Self {
        Self {
            gpci_data: HashMap::new(),
        }
    }

    /// Create with pre-populated data
    pub fn with_data(data: Vec<GpciData>) -> Self {
        let mut lookup = Self::new();
        for gpci in data {
            lookup.add_gpci_data(gpci);
        }
        lookup
    }

    /// Add GPCI data to the lookup
    pub fn add_gpci_data(&mut self, gpci: GpciData) {
        let key = (gpci.locality_code.clone(), gpci.year);
        self.gpci_data.insert(key, gpci);
    }

    /// Lookup GPCI data by locality code and year
    pub fn lookup(&self, locality_code: &str, year: i32) -> Result<&GpciData> {
        let key = (locality_code.to_string(), year);
        self.gpci_data
            .get(&key)
            .ok_or_else(|| Error::Validation(format!(
                "GPCI data not found for locality {} in year {}",
                locality_code, year
            )))
    }

    /// Check if GPCI data exists
    pub fn has_data(&self, locality_code: &str, year: i32) -> bool {
        let key = (locality_code.to_string(), year);
        self.gpci_data.contains_key(&key)
    }

    /// Get total count of GPCI entries
    pub fn count(&self) -> usize {
        self.gpci_data.len()
    }

    /// Load sample 2024 GPCI data for testing/demo
    pub fn load_sample_2024_data(&mut self) {
        // National average (locality 99 - used when specific locality not available)
        self.add_gpci_data(GpciData {
            locality_code: "99".to_string(),
            locality_name: "National Average".to_string(),
            year: 2024,
            work_gpci: Decimal::new(100, 2), // 1.00
            pe_gpci: Decimal::new(100, 2), // 1.00
            mp_gpci: Decimal::new(100, 2), // 1.00
        });

        // Manhattan, NY (high cost area)
        self.add_gpci_data(GpciData {
            locality_code: "01".to_string(),
            locality_name: "Manhattan, NY".to_string(),
            year: 2024,
            work_gpci: Decimal::new(10940, 4), // 1.094
            pe_gpci: Decimal::new(14950, 4), // 1.495
            mp_gpci: Decimal::new(16440, 4), // 1.644
        });

        // Queens, NY
        self.add_gpci_data(GpciData {
            locality_code: "02".to_string(),
            locality_name: "Queens, NY".to_string(),
            year: 2024,
            work_gpci: Decimal::new(10720, 4), // 1.072
            pe_gpci: Decimal::new(14140, 4), // 1.414
            mp_gpci: Decimal::new(15340, 4), // 1.534
        });

        // Los Angeles, CA
        self.add_gpci_data(GpciData {
            locality_code: "05".to_string(),
            locality_name: "Los Angeles, CA".to_string(),
            year: 2024,
            work_gpci: Decimal::new(10380, 4), // 1.038
            pe_gpci: Decimal::new(11790, 4), // 1.179
            mp_gpci: Decimal::new(8680, 4), // 0.868
        });

        // San Francisco, CA
        self.add_gpci_data(GpciData {
            locality_code: "09".to_string(),
            locality_name: "San Francisco, CA".to_string(),
            year: 2024,
            work_gpci: Decimal::new(10670, 4), // 1.067
            pe_gpci: Decimal::new(14860, 4), // 1.486
            mp_gpci: Decimal::new(8180, 4), // 0.818
        });

        // Rest of Texas (low cost area)
        self.add_gpci_data(GpciData {
            locality_code: "27".to_string(),
            locality_name: "Rest of Texas".to_string(),
            year: 2024,
            work_gpci: Decimal::new(9850, 4), // 0.985
            pe_gpci: Decimal::new(9250, 4), // 0.925
            mp_gpci: Decimal::new(9250, 4), // 0.925
        });

        // Dallas, TX
        self.add_gpci_data(GpciData {
            locality_code: "26".to_string(),
            locality_name: "Dallas, TX".to_string(),
            year: 2024,
            work_gpci: Decimal::new(10030, 4), // 1.003
            pe_gpci: Decimal::new(10070, 4), // 1.007
            mp_gpci: Decimal::new(10760, 4), // 1.076
        });

        // Chicago, IL
        self.add_gpci_data(GpciData {
            locality_code: "16".to_string(),
            locality_name: "Chicago, IL".to_string(),
            year: 2024,
            work_gpci: Decimal::new(10140, 4), // 1.014
            pe_gpci: Decimal::new(10740, 4), // 1.074
            mp_gpci: Decimal::new(12890, 4), // 1.289
        });

        // Boston, MA
        self.add_gpci_data(GpciData {
            locality_code: "13".to_string(),
            locality_name: "Boston, MA".to_string(),
            year: 2024,
            work_gpci: Decimal::new(10290, 4), // 1.029
            pe_gpci: Decimal::new(12230, 4), // 1.223
            mp_gpci: Decimal::new(8860, 4), // 0.886
        });

        // Miami, FL
        self.add_gpci_data(GpciData {
            locality_code: "03".to_string(),
            locality_name: "Miami, FL".to_string(),
            year: 2024,
            work_gpci: Decimal::new(9880, 4), // 0.988
            pe_gpci: Decimal::new(10510, 4), // 1.051
            mp_gpci: Decimal::new(25040, 4), // 2.504
        });
    }
}

impl Default for GpciLookup {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rvu_lookup_basic() {
        let mut lookup = RvuLookup::new();

        lookup.add_rvu_data(RvuData {
            hcpcs_code: "99213".to_string(),
            year: 2024,
            work_rvu: Decimal::new(110, 2),
            pe_rvu_facility: Decimal::new(48, 2),
            pe_rvu_non_facility: Decimal::new(113, 2),
            mp_rvu: Decimal::new(7, 2),
            total_rvu_facility: Decimal::new(165, 2),
            total_rvu_non_facility: Decimal::new(230, 2),
            global_period: Some("XXX".to_string()),
            pc_tc_indicator: None,
        });

        assert!(lookup.has_data("99213", 2024));
        assert!(!lookup.has_data("99213", 2023));
        assert!(!lookup.has_data("99214", 2024));
        assert_eq!(lookup.count(), 1);

        let result = lookup.lookup("99213", 2024);
        assert!(result.is_ok());
        let rvu = result.unwrap();
        assert_eq!(rvu.work_rvu, Decimal::new(110, 2));
    }

    #[test]
    fn test_rvu_lookup_not_found() {
        let lookup = RvuLookup::new();
        let result = lookup.lookup("99213", 2024);
        assert!(result.is_err());
    }

    #[test]
    fn test_rvu_lookup_sample_data() {
        let mut lookup = RvuLookup::new();
        lookup.load_sample_2024_em_data();

        assert!(lookup.has_data("99213", 2024));
        assert!(lookup.has_data("99214", 2024));
        assert!(lookup.has_data("99215", 2024));
        assert!(lookup.count() >= 9);

        let rvu_99213 = lookup.lookup("99213", 2024).unwrap();
        assert_eq!(rvu_99213.work_rvu, Decimal::new(110, 2));
        assert_eq!(rvu_99213.total_rvu_facility, Decimal::new(165, 2));
    }

    #[test]
    fn test_gpci_lookup_basic() {
        let mut lookup = GpciLookup::new();

        lookup.add_gpci_data(GpciData {
            locality_code: "01".to_string(),
            locality_name: "Manhattan, NY".to_string(),
            year: 2024,
            work_gpci: Decimal::new(10940, 4),
            pe_gpci: Decimal::new(14950, 4),
            mp_gpci: Decimal::new(16440, 4),
        });

        assert!(lookup.has_data("01", 2024));
        assert!(!lookup.has_data("01", 2023));
        assert_eq!(lookup.count(), 1);

        let result = lookup.lookup("01", 2024);
        assert!(result.is_ok());
        let gpci = result.unwrap();
        assert_eq!(gpci.work_gpci, Decimal::new(10940, 4));
    }

    #[test]
    fn test_gpci_lookup_sample_data() {
        let mut lookup = GpciLookup::new();
        lookup.load_sample_2024_data();

        assert!(lookup.has_data("01", 2024)); // Manhattan
        assert!(lookup.has_data("99", 2024)); // National average
        assert!(lookup.count() >= 9);

        let gpci_manhattan = lookup.lookup("01", 2024).unwrap();
        assert_eq!(gpci_manhattan.work_gpci, Decimal::new(10940, 4));

        let gpci_national = lookup.lookup("99", 2024).unwrap();
        assert_eq!(gpci_national.work_gpci, Decimal::new(100, 2));
    }
}
