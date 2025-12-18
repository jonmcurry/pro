use pro_parser_edi::EdiParser;

#[test]
fn test_multi_claim_hierarchy() {
    let mut parser = EdiParser::new();
    let result = parser.parse_file("c:/Users/jonmc/dev/pro/test_data/test_multi_claim_hierarchy.edi");

    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
    let transaction = result.unwrap();

    // Should have 10 claims total:
    // CLM*123, CLM*456, CLM*789, CLM*0011, CLM*1122, CLM*3344, CLM*5566, CLM*7788, CLM*9900, CLM*1010
    println!("Total claims parsed: {}", transaction.claims.len());

    let expected_pcns = vec!["123", "456", "789", "0011", "1122", "3344", "5566", "7788", "9900", "1010"];

    for (i, claim) in transaction.claims.iter().enumerate() {
        println!("Claim {}: PCN={}, subscriber_dob={:?}, patient_dob={:?}, patient_last={:?}, patient_first={:?}",
            i + 1,
            claim.patient_control_number,
            claim.subscriber_date_of_birth,
            claim.patient_date_of_birth,
            claim.patient_last_name,
            claim.patient_first_name);
    }

    assert_eq!(transaction.claims.len(), 10, "Expected 10 claims");

    // Verify all PCNs are present
    let parsed_pcns: Vec<&str> = transaction.claims.iter()
        .map(|c| c.patient_control_number.as_str())
        .collect();

    for expected_pcn in &expected_pcns {
        assert!(parsed_pcns.contains(expected_pcn), "Missing claim with PCN: {}", expected_pcn);
    }

    // Verify newborn claims (0011, 7788, 9900) have patient DOB
    for claim in &transaction.claims {
        if claim.patient_control_number == "0011" {
            assert!(claim.patient_date_of_birth.is_some(), "Claim 0011 should have patient DOB");
            assert_eq!(claim.patient_date_of_birth.unwrap().to_string(), "2023-12-10");
            assert_eq!(claim.patient_last_name, Some("TAYLOR".to_string()));
            assert_eq!(claim.patient_first_name, Some("BABY".to_string()));
        }
        if claim.patient_control_number == "7788" {
            assert!(claim.patient_date_of_birth.is_some(), "Claim 7788 should have patient DOB");
            assert_eq!(claim.patient_date_of_birth.unwrap().to_string(), "2023-12-15");
            assert_eq!(claim.patient_last_name, Some("WHITE".to_string()));
            assert_eq!(claim.patient_first_name, Some("NEWBORN".to_string()));
        }
        if claim.patient_control_number == "9900" {
            assert!(claim.patient_date_of_birth.is_some(), "Claim 9900 should have patient DOB");
            assert_eq!(claim.patient_date_of_birth.unwrap().to_string(), "2023-12-15");
            assert_eq!(claim.patient_last_name, Some("WHITE".to_string()));
            assert_eq!(claim.patient_first_name, Some("NEWBORN".to_string()));
        }
    }

    println!("All claims parsed correctly!");
}
