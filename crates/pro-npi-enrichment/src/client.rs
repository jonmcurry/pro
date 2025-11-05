use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::debug;

const NPI_REGISTRY_BASE_URL: &str = "https://npiregistry.cms.hhs.gov/api/";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
const USER_AGENT: &str = "ProfessionalSMART/1.6.1 (Healthcare Claims Processing)";

/// NPI Registry API Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpiRegistryResponse {
    pub result_count: i32,
    pub results: Vec<NpiProvider>,
}

/// Provider information from NPI Registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NpiProvider {
    pub number: String,
    pub enumeration_type: String,
    pub basic: BasicInfo,
    pub taxonomies: Vec<Taxonomy>,
    pub addresses: Vec<Address>,
    #[serde(default)]
    pub other_names: Vec<OtherName>,
    #[serde(default)]
    pub identifiers: Vec<Identifier>,
}

/// Basic provider demographic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicInfo {
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub middle_name: Option<String>,
    pub credential: Option<String>,
    pub sole_proprietor: Option<String>,
    pub gender: Option<String>,
    pub enumeration_date: Option<String>,
    pub last_updated: Option<String>,
    pub status: Option<String>,
    pub name: Option<String>,
    pub authorized_official_first_name: Option<String>,
    pub authorized_official_last_name: Option<String>,
    pub authorized_official_middle_name: Option<String>,
    pub authorized_official_title_or_position: Option<String>,
    pub authorized_official_telephone_number: Option<String>,
    pub organization_name: Option<String>,
}

/// Provider taxonomy (specialty) information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Taxonomy {
    pub code: String,
    pub taxonomy_group: Option<String>,
    pub desc: String,
    pub primary: bool,
    pub state: Option<String>,
    pub license: Option<String>,
}

/// Provider address information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Address {
    pub country_code: Option<String>,
    pub country_name: Option<String>,
    pub address_purpose: String,
    pub address_type: Option<String>,
    pub address_1: Option<String>,
    pub address_2: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub postal_code: Option<String>,
    pub telephone_number: Option<String>,
    pub fax_number: Option<String>,
}

/// Other names (aliases, previous names)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OtherName {
    #[serde(rename = "type")]
    pub name_type: Option<String>,
    pub code: Option<String>,
    pub credential: Option<String>,
    pub first_name: Option<String>,
    pub last_name: Option<String>,
    pub middle_name: Option<String>,
    pub prefix: Option<String>,
    pub suffix: Option<String>,
}

/// Additional identifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identifier {
    pub code: Option<String>,
    pub desc: Option<String>,
    pub identifier: Option<String>,
    pub state: Option<String>,
    pub issuer: Option<String>,
}

/// NPI Registry API Client
pub struct NpiRegistryClient {
    client: reqwest::Client,
    base_url: String,
}

impl NpiRegistryClient {
    /// Create a new NPI Registry API client
    pub fn new() -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .user_agent(USER_AGENT)
            .build()
            .context("Failed to build HTTP client for NPI Registry")?;

        Ok(Self {
            client,
            base_url: NPI_REGISTRY_BASE_URL.to_string(),
        })
    }

    /// Lookup provider by NPI number
    ///
    /// # Arguments
    /// * `npi` - 10-digit National Provider Identifier
    ///
    /// # Returns
    /// * `Ok(NpiRegistryResponse)` - Provider information from NPI Registry
    /// * `Err` - API error, network error, or NPI not found
    ///
    /// # Example
    /// ```no_run
    /// # use pro_npi_enrichment::client::NpiRegistryClient;
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// let client = NpiRegistryClient::new()?;
    /// let response = client.lookup_npi("1234567890").await?;
    /// println!("Found provider: {:?}", response.results[0].basic.name);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn lookup_npi(&self, npi: &str) -> Result<NpiRegistryResponse> {
        // Validate NPI format
        if npi.len() != 10 || !npi.chars().all(|c| c.is_ascii_digit()) {
            anyhow::bail!("Invalid NPI format: {} (expected 10 digits)", npi);
        }

        // Build API URL
        let url = format!("{}?version=2.1&number={}", self.base_url, npi);

        debug!("Calling NPI Registry API: {}", url);

        // Make HTTP request
        let response = self.client
            .get(&url)
            .send()
            .await
            .context("Failed to send NPI Registry API request")?;

        // Check HTTP status
        let status = response.status();
        if !status.is_success() {
            let error_body = response.text().await.unwrap_or_else(|_| String::from("(unable to read error body)"));
            anyhow::bail!("NPI Registry API returned error: {} - {}", status, error_body);
        }

        // Parse JSON response
        let data: NpiRegistryResponse = response
            .json()
            .await
            .context("Failed to parse NPI Registry API response as JSON")?;

        // Check if NPI was found
        if data.result_count == 0 {
            anyhow::bail!("NPI not found in registry: {}", npi);
        }

        if data.results.is_empty() {
            anyhow::bail!("NPI Registry returned 0 results for NPI: {}", npi);
        }

        debug!("Successfully retrieved data for NPI: {} (result_count: {})", npi, data.result_count);

        Ok(data)
    }

    /// Lookup multiple NPIs in a single batch (if API supports it in the future)
    /// Currently calls lookup_npi for each NPI individually
    pub async fn lookup_npis_batch(&self, npis: &[String]) -> Vec<Result<NpiRegistryResponse>> {
        let mut results = Vec::with_capacity(npis.len());

        for npi in npis {
            results.push(self.lookup_npi(npi).await);
        }

        results
    }
}

impl Default for NpiRegistryClient {
    fn default() -> Self {
        Self::new().expect("Failed to create default NPI Registry client")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npi_validation() {
        let client = NpiRegistryClient::new().unwrap();

        // Valid NPI format
        assert!(matches!(
            tokio_test::block_on(client.lookup_npi("1234567890")),
            Err(_) // Will fail because it's not a real NPI, but format is valid
        ));

        // Invalid format - too short
        let result = tokio_test::block_on(client.lookup_npi("123"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid NPI format"));

        // Invalid format - contains letters
        let result = tokio_test::block_on(client.lookup_npi("123ABC7890"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Invalid NPI format"));
    }

    #[test]
    fn test_client_creation() {
        let client = NpiRegistryClient::new();
        assert!(client.is_ok());

        let client = client.unwrap();
        assert_eq!(client.base_url, NPI_REGISTRY_BASE_URL);
    }
}
