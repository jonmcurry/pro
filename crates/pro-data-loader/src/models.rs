//! Data models for master data entities

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Organization {
    pub organization_code: String,
    pub organization_name: String,
    pub tax_id: Option<String>,
    pub contact_email: Option<String>,
    pub address_line1: Option<String>,
    pub address_line2: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub zip_code: Option<String>,
    #[serde(default = "default_true")]
    pub active: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Region {
    pub organization_code: String,
    pub region_code: String,
    pub region_name: String,
    pub manager_name: Option<String>,
    pub manager_email: Option<String>,
    #[serde(default = "default_true")]
    pub active: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Facility {
    pub organization_code: String,
    pub region_code: String,
    pub facility_code: String,
    pub facility_name: String,
    #[serde(alias = "npi")]
    pub facility_npi: Option<String>,
    pub tax_id: Option<String>,
    pub address_line1: Option<String>,
    pub address_line2: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub zip_code: Option<String>,
    pub phone: Option<String>,
    pub ehr_system: Option<String>,
    #[serde(default = "default_true")]
    pub active: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Provider {
    pub facility_code: String,
    pub provider_npi: String,
    #[serde(alias = "physician_first_name", alias = "doctor_first_name")]
    pub first_name: String,
    pub last_name: String,
    pub middle_name: Option<String>,
    #[serde(alias = "full_physician_name", alias = "physician_name", alias = "doctor_name", alias = "provider_name")]
    pub full_name: Option<String>,
    pub credentials: Option<String>,
    pub specialty: Option<String>,
    pub taxonomy_code: Option<String>,
    pub email: Option<String>,
    pub phone: Option<String>,
    #[serde(default = "default_true")]
    pub active: bool,
}

// Helper structs for database operations
#[derive(Debug, Clone)]
pub struct OrganizationDb {
    pub organization_id: i64,
    pub organization_code: String,
}

#[derive(Debug, Clone)]
pub struct RegionDb {
    pub region_id: i64,
    pub organization_id: i64,
    pub region_code: String,
}

#[derive(Debug, Clone)]
pub struct FacilityDb {
    pub facility_id: i64,
    pub region_id: i64,
    pub facility_code: String,
}

fn default_true() -> bool {
    true
}

// Import results
#[derive(Debug, Default, Clone)]
pub struct ImportResults {
    pub organizations_inserted: usize,
    pub regions_inserted: usize,
    pub facilities_inserted: usize,
    pub providers_inserted: usize,
    pub errors: Vec<String>,
}

impl ImportResults {
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn total_inserted(&self) -> usize {
        self.organizations_inserted + self.regions_inserted + self.facilities_inserted + self.providers_inserted
    }
}
