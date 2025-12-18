use pro_parser_edi::EdiParser;

#[test]
fn test_newborn_patient_dob() {
    let mut parser = EdiParser::new();
    let result = parser.parse_file("c:/Users/jonmc/dev/pro/test_data/test_newborn.edi");

    assert!(result.is_ok(), "Failed to parse: {:?}", result.err());
    let transaction = result.unwrap();

    assert_eq!(transaction.claims.len(), 1, "Expected 1 claim");
    let claim = &transaction.claims[0];

    println!("Claim PCN: {}", claim.patient_control_number);
    println!("Subscriber DOB: {:?}", claim.subscriber_date_of_birth);
    println!("Patient DOB: {:?}", claim.patient_date_of_birth);
    println!("Patient Last Name: {:?}", claim.patient_last_name);
    println!("Patient First Name: {:?}", claim.patient_first_name);

    assert_eq!(claim.patient_control_number, "NEWBORN001");

    // Subscriber should be JANE SMITH born 1990-01-15
    assert!(claim.subscriber_date_of_birth.is_some(), "subscriber_date_of_birth should be set");
    assert_eq!(claim.subscriber_date_of_birth.unwrap().to_string(), "1990-01-15");

    // Patient should be BABY BOY SMITH born 2023-12-10
    assert!(claim.patient_date_of_birth.is_some(), "patient_date_of_birth should be set");
    assert_eq!(claim.patient_date_of_birth.unwrap().to_string(), "2023-12-10");
    assert_eq!(claim.patient_last_name, Some("SMITH".to_string()));
    assert_eq!(claim.patient_first_name, Some("BABY".to_string()));
}
