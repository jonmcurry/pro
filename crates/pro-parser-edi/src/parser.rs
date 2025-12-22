// Main 837P EDI parser

use crate::loops::*;
use crate::segments::*;
use crate::types::*;
use crate::validator;
use pro_common::{Error, Result};
use std::collections::HashMap;

/// EDI Parser for 837P Professional Claims
pub struct EdiParser {
    element_separator: char,
    segment_terminator: char,
    component_separator: char,
}

impl Default for EdiParser {
    fn default() -> Self {
        Self::new()
    }
}

impl EdiParser {
    /// Create a new EDI parser with default delimiters
    pub fn new() -> Self {
        Self {
            element_separator: '*',
            segment_terminator: '~',
            component_separator: ':',
        }
    }

    /// Parse complete 837P transaction from EDI string
    pub fn parse(&mut self, edi_content: &str) -> Result<Transaction837p> {
        // Extract delimiters from ISA segment
        self.extract_delimiters(edi_content)?;

        // Split into segments
        let segments = self.split_segments(edi_content)?;

        // Validate basic structure
        validator::validate_transaction_structure(&segments)?;

        // Parse header segments
        let isa = self.parse_isa(&segments)?;
        let gs = self.parse_gs(&segments)?;
        let st = self.parse_st(&segments)?;

        // Find loop boundaries
        let loop_map = self.identify_loops(&segments)?;

        // Parse loops
        let submitter = if let Some(submitter_segments) = loop_map.get("1000A") {
            parse_submitter(submitter_segments)?
        } else {
            return Err(Error::EdiParse("Missing Loop 1000A (Submitter)".to_string()));
        };

        let receiver = if let Some(receiver_segments) = loop_map.get("1000B") {
            parse_receiver(receiver_segments)?
        } else {
            return Err(Error::EdiParse("Missing Loop 1000B (Receiver)".to_string()));
        };

        let billing_provider = if let Some(provider_segments) = loop_map.get("2000A") {
            parse_billing_provider(provider_segments)?
        } else {
            return Err(Error::EdiParse("Missing Loop 2000A (Billing Provider)".to_string()));
        };

        // Parse all claims (Loop 2000B/2300)
        let claims = self.parse_claims(&segments)?;

        Ok(Transaction837p {
            interchange_control_header: isa,
            functional_group_header: gs,
            transaction_set_header: st,
            submitter,
            receiver,
            billing_provider,
            claims,
        })
    }

    /// Extract delimiters from ISA segment
    fn extract_delimiters(&mut self, edi_content: &str) -> Result<()> {
        if edi_content.len() < 106 {
            return Err(Error::EdiParse("EDI content too short to contain ISA segment".to_string()));
        }

        // ISA segment has fixed positions for delimiters
        self.element_separator = edi_content.chars().nth(3)
            .ok_or_else(|| Error::EdiParse("Cannot extract element separator".to_string()))?;

        self.component_separator = edi_content.chars().nth(104)
            .ok_or_else(|| Error::EdiParse("Cannot extract component separator".to_string()))?;

        // Find segment terminator (should be after 106th character)
        self.segment_terminator = edi_content.chars().nth(105)
            .ok_or_else(|| Error::EdiParse("Cannot extract segment terminator".to_string()))?;

        Ok(())
    }

    /// Split EDI content into segments
    fn split_segments(&self, edi_content: &str) -> Result<Vec<EdiSegment>> {
        let mut segments = Vec::new();
        let segment_strings: Vec<&str> = edi_content
            .split(self.segment_terminator)
            .filter(|s| !s.trim().is_empty())
            .collect();

        for segment_str in segment_strings {
            let parts: Vec<&str> = segment_str.split(self.element_separator).collect();
            if parts.is_empty() {
                continue;
            }

            let segment_id = parts[0].trim().to_string();
            let elements: Vec<String> = parts[1..]
                .iter()
                .map(|e| e.trim().to_string())
                .collect();

            segments.push(EdiSegment {
                segment_id,
                elements,
            });
        }

        Ok(segments)
    }

    /// Parse ISA segment
    fn parse_isa(&self, segments: &[EdiSegment]) -> Result<InterchangeControlHeader> {
        let isa = segments
            .iter()
            .find(|s| s.segment_id == "ISA")
            .ok_or_else(|| Error::EdiParse("Missing ISA segment".to_string()))?;

        Ok(InterchangeControlHeader {
            authorization_info_qualifier: isa.get_or_empty(0).to_string(),
            authorization_information: isa.get_or_empty(1).to_string(),
            security_info_qualifier: isa.get_or_empty(2).to_string(),
            security_information: isa.get_or_empty(3).to_string(),
            interchange_id_qualifier: isa.get_or_empty(4).to_string(),
            interchange_sender_id: isa.get_or_empty(5).to_string(),
            interchange_id_qualifier_2: isa.get_or_empty(6).to_string(),
            interchange_receiver_id: isa.get_or_empty(7).to_string(),
            interchange_date: isa.get_or_empty(8).to_string(),
            interchange_time: isa.get_or_empty(9).to_string(),
            repetition_separator: isa.get_or_empty(10).chars().next().unwrap_or('^'),
            interchange_control_version: isa.get_or_empty(11).to_string(),
            interchange_control_number: isa.get_or_empty(12).to_string(),
            acknowledgment_requested: isa.get_or_empty(13).to_string(),
            usage_indicator: isa.get_or_empty(14).to_string(),
            component_element_separator: self.component_separator,
        })
    }

    /// Parse GS segment
    fn parse_gs(&self, segments: &[EdiSegment]) -> Result<FunctionalGroupHeader> {
        let gs = segments
            .iter()
            .find(|s| s.segment_id == "GS")
            .ok_or_else(|| Error::EdiParse("Missing GS segment".to_string()))?;

        Ok(FunctionalGroupHeader {
            functional_identifier_code: gs.get_or_empty(0).to_string(),
            application_sender_code: gs.get_or_empty(1).to_string(),
            application_receiver_code: gs.get_or_empty(2).to_string(),
            date: gs.get_or_empty(3).to_string(),
            time: gs.get_or_empty(4).to_string(),
            group_control_number: gs.get_or_empty(5).to_string(),
            responsible_agency_code: gs.get_or_empty(6).to_string(),
            version_release: gs.get_or_empty(7).to_string(),
        })
    }

    /// Parse ST segment
    fn parse_st(&self, segments: &[EdiSegment]) -> Result<TransactionSetHeader> {
        let st = segments
            .iter()
            .find(|s| s.segment_id == "ST")
            .ok_or_else(|| Error::EdiParse("Missing ST segment".to_string()))?;

        Ok(TransactionSetHeader {
            transaction_set_identifier_code: st.get_or_empty(0).to_string(),
            transaction_set_control_number: st.get_or_empty(1).to_string(),
            implementation_convention_reference: st.get_or_empty(2).to_string(),
        })
    }

    /// Identify loop boundaries in the segment list
    fn identify_loops(&self, segments: &[EdiSegment]) -> Result<HashMap<String, Vec<EdiSegment>>> {
        use tracing::debug;

        let mut loop_map: HashMap<String, Vec<EdiSegment>> = HashMap::new();
        let mut current_loop: Option<String> = None;
        let mut loop_segments: Vec<EdiSegment> = Vec::new();

        debug!("[LOOP_DEBUG] Starting loop identification for {} segments", segments.len());

        for segment in segments {
            match segment.segment_id.as_str() {
                "NM1" => {
                    // Only process NM1 as loop starter if we're NOT inside a hierarchical loop (2000A/2000B/2000C)
                    // NM1*41 and NM1*40 appear BEFORE any HL segments
                    // NM1*85 and others should stay within their parent hierarchical loop
                    let in_hierarchical_loop = current_loop.as_ref()
                        .map(|l| l.starts_with("2000") || l.starts_with("2010") || l.starts_with("2310"))
                        .unwrap_or(false);

                    if !in_hierarchical_loop {
                        if let Some(entity_id) = segment.get(0) {
                            let new_loop = match entity_id {
                                "41" => Some("1000A".to_string()), // Submitter
                                "40" => Some("1000B".to_string()), // Receiver
                                _ => None,
                            };

                            if let Some(loop_name) = new_loop {
                                // Save previous loop if any
                                if let Some(prev_loop) = current_loop.take() {
                                    loop_map.insert(prev_loop, loop_segments.clone());
                                    loop_segments.clear();
                                }
                                current_loop = Some(loop_name);
                            }
                        }
                    }
                    // NM1*85, *IL, *PR, etc. will be included in the current hierarchical loop's segments
                }
                "HL" => {
                    // Hierarchical level indicates new loop
                    debug!("[LOOP_DEBUG] Found HL segment with {} elements: {:?}", segment.elements.len(), segment.elements);
                    if let Some(level_code) = segment.get(2) {
                        debug!("[LOOP_DEBUG] HL level_code = '{}'", level_code);
                        let new_loop = match level_code {
                            "20" => Some("2000A".to_string()), // Billing Provider Level
                            "22" => Some("2000B".to_string()), // Subscriber Level
                            "23" => Some("2000C".to_string()), // Patient Level (if different from subscriber)
                            _ => None,
                        };

                        if let Some(loop_name) = new_loop {
                            debug!("[LOOP_DEBUG] Starting loop: {}", loop_name);
                            if let Some(prev_loop) = current_loop.take() {
                                debug!("[LOOP_DEBUG] Saving previous loop: {} with {} segments", prev_loop, loop_segments.len());
                                loop_map.insert(prev_loop, loop_segments.clone());
                                loop_segments.clear();
                            }
                            current_loop = Some(loop_name);
                        } else {
                            debug!("[LOOP_DEBUG] HL level_code '{}' does not match any known loop", level_code);
                        }
                    }
                }
                _ => {}
            }

            if current_loop.is_some() {
                loop_segments.push(segment.clone());
            }
        }

        // Save last loop
        if let Some(loop_name) = current_loop {
            debug!("[LOOP_DEBUG] Saving final loop: {} with {} segments", loop_name, loop_segments.len());
            loop_map.insert(loop_name, loop_segments);
        }

        debug!("[LOOP_DEBUG] Loop identification complete. Found loops: {:?}", loop_map.keys().collect::<Vec<_>>());

        // Log if Loop 2000A is missing
        if !loop_map.contains_key("2000A") {
            debug!("[LOOP_DEBUG] WARNING: Loop 2000A (Billing Provider Level) was NOT found!");
        }

        Ok(loop_map)
    }

    /// Parse all claims from the transaction
    fn parse_claims(&self, segments: &[EdiSegment]) -> Result<Vec<ParsedClaim>> {
        let mut claims = Vec::new();
        let mut current_claim_segments = Vec::new();
        let mut subscriber_segments: Vec<EdiSegment> = Vec::new(); // Segments from HL*22 (subscriber level)
        let mut in_subscriber_loop = false;
        let mut in_patient_loop = false;
        let mut in_claim = false;

        // Extract BHT segment to get billing_date (transaction creation date)
        // BHT appears once per transaction, before the HL loops
        let billing_date = segments.iter()
            .find(|s| s.segment_id == "BHT")
            .and_then(|bht_seg| BhtSegment::parse(bht_seg).ok())
            .and_then(|bht| bht.transaction_date);

        for segment in segments {
            match segment.segment_id.as_str() {
                "HL" => {
                    // HL segment format: HL*id*parent_id*level_code*child_code
                    if let Some(level_code) = segment.get_optional(2) {
                        match level_code.as_str() {
                            "22" => {
                                // Subscriber level (2000B) - contains NM1*IL, NM1*PR, SBR, etc.
                                // IMPORTANT: Finalize any in-progress claim before starting new subscriber
                                if in_claim && !current_claim_segments.is_empty() {
                                    let mut claim = parse_claim_info(&current_claim_segments)?;
                                    claim.billing_date = billing_date;
                                    claims.push(claim);
                                    current_claim_segments.clear();
                                }
                                // Reset subscriber segments for new subscriber
                                subscriber_segments.clear();
                                in_subscriber_loop = true;
                                in_patient_loop = false;
                                in_claim = false;
                            }
                            "23" => {
                                // Patient level (2000C) - claims follow under this
                                // IMPORTANT: Finalize any in-progress claim before starting patient loop
                                if in_claim && !current_claim_segments.is_empty() {
                                    let mut claim = parse_claim_info(&current_claim_segments)?;
                                    claim.billing_date = billing_date;
                                    claims.push(claim);
                                    current_claim_segments.clear();
                                    in_claim = false;
                                }
                                in_subscriber_loop = false;
                                in_patient_loop = true;
                            }
                            _ => {}
                        }
                    }
                }
                "CLM" => {
                    // Start of new claim
                    if in_claim && !current_claim_segments.is_empty() {
                        // Parse previous claim
                        let mut claim = parse_claim_info(&current_claim_segments)?;
                        claim.billing_date = billing_date; // Set billing date from BHT segment
                        claims.push(claim);
                        current_claim_segments.clear();
                    }
                    in_claim = true;
                    in_subscriber_loop = false;
                    // Prepend subscriber-level segments (NM1*PR, NM1*IL, SBR, etc.) to this claim
                    current_claim_segments = subscriber_segments.clone();
                    current_claim_segments.push(segment.clone());
                }
                "SE" => {
                    // End of transaction set - finish last claim
                    if in_claim && !current_claim_segments.is_empty() {
                        let mut claim = parse_claim_info(&current_claim_segments)?;
                        claim.billing_date = billing_date; // Set billing date from BHT segment
                        claims.push(claim);
                        current_claim_segments.clear();
                    }
                    in_claim = false;
                    in_subscriber_loop = false;
                    in_patient_loop = false;
                }
                // Collect subscriber-level segments (SBR, NM1, DMG, etc.) that appear before CLM
                "SBR" | "DMG" | "NM1" | "N3" | "N4" | "REF" | "PER" | "PAT" => {
                    if in_subscriber_loop && !in_claim {
                        // Subscriber-level segment (Loop 2000B/2010BA/2010BB) - save for all claims under this subscriber
                        subscriber_segments.push(segment.clone());
                    } else if in_patient_loop && !in_claim {
                        // Patient-level segment (Loop 2000C/2010CA) - also save to subscriber segments
                        // These apply to claims under this patient
                        subscriber_segments.push(segment.clone());
                    } else if in_claim {
                        // Already in claim - collect it (claim-level NM1 like NM1*82 rendering provider)
                        current_claim_segments.push(segment.clone());
                    }
                }
                _ => {
                    if in_claim {
                        current_claim_segments.push(segment.clone());
                    }
                }
            }
        }

        Ok(claims)
    }

    /// Parse a file from disk
    pub fn parse_file(&mut self, file_path: &str) -> Result<Transaction837p> {
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| Error::Io(e))?;
        self.parse(&content)
    }

    /// PHASE 5: Parse claims as a stream for real-time processing
    ///
    /// This method parses the EDI file and emits claims one-at-a-time as they're
    /// encountered, allowing for streaming processing with lower memory usage.
    ///
    /// Returns a stream of `Result<ParsedClaim>` that yields claims in the order
    /// they appear in the file (maintaining FIFO compliance).
    pub fn parse_stream(
        &mut self,
        edi_content: String,
    ) -> impl futures_core::Stream<Item = Result<ParsedClaim>> + '_ {
        async_stream::stream! {
            // Extract delimiters from ISA segment
            if let Err(e) = self.extract_delimiters(&edi_content) {
                yield Err(e);
                return;
            }

            // Split into segments
            let segments = match self.split_segments(&edi_content) {
                Ok(s) => s,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            // Validate basic structure
            if let Err(e) = validator::validate_transaction_structure(&segments) {
                yield Err(e);
                return;
            }

            // Extract BHT segment to get billing_date (transaction creation date)
            let billing_date = segments.iter()
                .find(|s| s.segment_id == "BHT")
                .and_then(|bht_seg| BhtSegment::parse(bht_seg).ok())
                .and_then(|bht| bht.transaction_date);

            // Stream claims one at a time
            let mut current_claim_segments = Vec::new();
            let mut subscriber_segments: Vec<EdiSegment> = Vec::new(); // Segments from HL*22 (subscriber level)
            let mut in_subscriber_loop = false;
            let mut in_patient_loop = false;
            let mut in_claim = false;

            for segment in segments {
                match segment.segment_id.as_str() {
                    "HL" => {
                        // HL segment format: HL*id*parent_id*level_code*child_code
                        if let Some(level_code) = segment.get_optional(2) {
                            match level_code.as_str() {
                                "22" => {
                                    // Subscriber level (2000B) - contains NM1*IL, NM1*PR, SBR, etc.
                                    // Reset subscriber segments for new subscriber
                                    subscriber_segments.clear();
                                    in_subscriber_loop = true;
                                    in_patient_loop = false;
                                    in_claim = false;
                                }
                                "23" => {
                                    // Patient level (2000C) - claims follow under this
                                    in_subscriber_loop = false;
                                    in_patient_loop = true;
                                }
                                _ => {}
                            }
                        }
                    }
                    "CLM" => {
                        // Start of new claim
                        if in_claim && !current_claim_segments.is_empty() {
                            // Emit previous claim
                            match parse_claim_info(&current_claim_segments) {
                                Ok(mut claim) => {
                                    claim.billing_date = billing_date; // Set billing date from BHT segment
                                    yield Ok(claim)
                                },
                                Err(e) => yield Err(e),
                            }
                            current_claim_segments.clear();
                        }
                        in_claim = true;
                        in_subscriber_loop = false;
                        // Prepend subscriber-level segments (NM1*PR, NM1*IL, SBR, etc.) to this claim
                        current_claim_segments = subscriber_segments.clone();
                        current_claim_segments.push(segment.clone());
                    }
                    "SE" => {
                        // End of transaction set - emit last claim
                        if in_claim && !current_claim_segments.is_empty() {
                            match parse_claim_info(&current_claim_segments) {
                                Ok(mut claim) => {
                                    claim.billing_date = billing_date; // Set billing date from BHT segment
                                    yield Ok(claim)
                                },
                                Err(e) => yield Err(e),
                            }
                            current_claim_segments.clear();
                        }
                        in_claim = false;
                        in_subscriber_loop = false;
                        in_patient_loop = false;
                    }
                    // Collect subscriber-level segments (SBR, NM1, DMG, etc.) that appear before CLM
                    "SBR" | "DMG" | "NM1" | "N3" | "N4" | "REF" | "PER" | "PAT" => {
                        if in_subscriber_loop && !in_claim {
                            // Subscriber-level segment (Loop 2000B/2010BA/2010BB) - save for all claims under this subscriber
                            subscriber_segments.push(segment.clone());
                        } else if in_patient_loop && !in_claim {
                            // Patient-level segment (Loop 2000C/2010CA) - also save to subscriber segments
                            subscriber_segments.push(segment.clone());
                        } else if in_claim {
                            // Already in claim - collect it (claim-level NM1 like NM1*82 rendering provider)
                            current_claim_segments.push(segment.clone());
                        }
                    }
                    _ => {
                        if in_claim {
                            current_claim_segments.push(segment.clone());
                        }
                    }
                }
            }
        }
    }

    /// PHASE 5: Parse file from disk as a stream
    pub async fn parse_file_stream(
        &mut self,
        file_path: &str,
    ) -> Result<impl futures_core::Stream<Item = Result<ParsedClaim>> + '_> {
        let content = tokio::fs::read_to_string(file_path)
            .await
            .map_err(|e| Error::Io(e))?;
        Ok(self.parse_stream(content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_delimiters() {
        let isa = "ISA*00*          *00*          *ZZ*SUBMITTERID    *ZZ*RECEIVERID     *240115*1430*^*00501*000000001*0*P*:~";
        let mut parser = EdiParser::new();
        parser.extract_delimiters(isa).unwrap();

        assert_eq!(parser.element_separator, '*');
        assert_eq!(parser.segment_terminator, '~');
        assert_eq!(parser.component_separator, ':');
    }

    #[test]
    fn test_split_segments() {
        let edi = "ISA*00*TEST~GS*HC*SENDER*RECEIVER~ST*837*0001~";
        let mut parser = EdiParser::new();
        parser.extract_delimiters(edi).unwrap();
        let segments = parser.split_segments(edi).unwrap();

        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0].segment_id, "ISA");
        assert_eq!(segments[1].segment_id, "GS");
        assert_eq!(segments[2].segment_id, "ST");
    }
}
