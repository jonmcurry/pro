// Medicare payment calculation with RVU, GPCI, and modifier adjustments

use crate::lookup::{GpciLookup, RvuLookup};
use crate::types::{ConversionFactor, ModifierAdjustment, PlaceOfService};
use pro_common::{Error, Result};
use rust_decimal::Decimal;

/// Payment calculation result
#[derive(Debug, Clone)]
pub struct PaymentCalculation {
    /// HCPCS/CPT code
    pub hcpcs_code: String,
    /// Calendar year
    pub year: i32,
    /// Locality code
    pub locality_code: String,
    /// Place of service code
    pub pos_code: String,
    /// Is facility setting
    pub is_facility: bool,

    // RVU components
    pub work_rvu: Decimal,
    pub pe_rvu: Decimal,
    pub mp_rvu: Decimal,
    pub total_rvu: Decimal,

    // GPCI adjustments
    pub work_gpci: Decimal,
    pub pe_gpci: Decimal,
    pub mp_gpci: Decimal,

    // Geographic-adjusted RVUs
    pub adjusted_work_rvu: Decimal,
    pub adjusted_pe_rvu: Decimal,
    pub adjusted_mp_rvu: Decimal,
    pub adjusted_total_rvu: Decimal,

    // Payment calculation
    pub conversion_factor: Decimal,
    pub base_payment: Decimal,

    // Modifier adjustments
    pub modifiers: Vec<String>,
    pub modifier_adjustment_percent: Decimal,
    pub modifier_adjustment_description: Option<String>,

    // Final payment
    pub final_payment: Decimal,
    pub units: Decimal,
    pub total_payment: Decimal,
}

impl PaymentCalculation {
    /// Format payment as currency string
    pub fn format_payment(&self) -> String {
        format!("${:.2}", self.total_payment)
    }

    /// Get payment per unit
    pub fn per_unit_payment(&self) -> Decimal {
        self.final_payment
    }
}

/// Medicare payment calculator
pub struct PaymentCalculator {
    rvu_lookup: RvuLookup,
    gpci_lookup: GpciLookup,
}

impl PaymentCalculator {
    /// Create new payment calculator
    pub fn new(rvu_lookup: RvuLookup, gpci_lookup: GpciLookup) -> Self {
        Self {
            rvu_lookup,
            gpci_lookup,
        }
    }

    /// Create calculator with sample 2024 data
    pub fn with_sample_data() -> Self {
        let mut rvu_lookup = RvuLookup::new();
        rvu_lookup.load_sample_2024_em_data();

        let mut gpci_lookup = GpciLookup::new();
        gpci_lookup.load_sample_2024_data();

        Self::new(rvu_lookup, gpci_lookup)
    }

    /// Calculate Medicare payment for a service
    ///
    /// Formula: Payment = [(Work RVU × Work GPCI) + (PE RVU × PE GPCI) + (MP RVU × MP GPCI)] × CF × Modifier Adjustment × Units
    pub fn calculate(
        &self,
        hcpcs_code: &str,
        year: i32,
        locality_code: &str,
        pos_code: &str,
        modifiers: Vec<String>,
        units: Decimal,
    ) -> Result<PaymentCalculation> {
        // Lookup RVU data
        let rvu = self.rvu_lookup.lookup(hcpcs_code, year)?;

        // Lookup GPCI data
        let gpci = self.gpci_lookup.lookup(locality_code, year)?;

        // Get conversion factor
        let cf = ConversionFactor::for_year(year)
            .ok_or_else(|| Error::Validation(format!("Conversion factor not available for year {}", year)))?;

        // Determine if facility or non-facility
        let is_facility = PlaceOfService::is_facility(pos_code);

        // Get appropriate RVU values
        let work_rvu = rvu.work_rvu;
        let pe_rvu = rvu.pe_rvu(is_facility);
        let mp_rvu = rvu.mp_rvu;
        let total_rvu = work_rvu + pe_rvu + mp_rvu;

        // Apply GPCI adjustments to each RVU component
        let adjusted_work_rvu = work_rvu * gpci.work_gpci;
        let adjusted_pe_rvu = pe_rvu * gpci.pe_gpci;
        let adjusted_mp_rvu = mp_rvu * gpci.mp_gpci;
        let adjusted_total_rvu = adjusted_work_rvu + adjusted_pe_rvu + adjusted_mp_rvu;

        // Calculate base payment (before modifiers)
        let base_payment = adjusted_total_rvu * cf.factor;

        // Apply modifier adjustments
        let (modifier_percent, modifier_desc) = self.calculate_modifier_adjustment(&modifiers)?;

        // Calculate final payment per unit
        let final_payment = base_payment * modifier_percent;

        // Calculate total payment (with units)
        let total_payment = final_payment * units;

        Ok(PaymentCalculation {
            hcpcs_code: hcpcs_code.to_string(),
            year,
            locality_code: locality_code.to_string(),
            pos_code: pos_code.to_string(),
            is_facility,
            work_rvu,
            pe_rvu,
            mp_rvu,
            total_rvu,
            work_gpci: gpci.work_gpci,
            pe_gpci: gpci.pe_gpci,
            mp_gpci: gpci.mp_gpci,
            adjusted_work_rvu,
            adjusted_pe_rvu,
            adjusted_mp_rvu,
            adjusted_total_rvu,
            conversion_factor: cf.factor,
            base_payment,
            modifiers: modifiers.clone(),
            modifier_adjustment_percent: modifier_percent,
            modifier_adjustment_description: modifier_desc,
            final_payment,
            units,
            total_payment,
        })
    }

    /// Calculate modifier adjustment percentage
    ///
    /// Returns (adjustment_percent, description)
    /// - If no payment-affecting modifiers: (1.00, None)
    /// - If single modifier: applies that modifier's percentage
    /// - If multiple modifiers: applies first applicable modifier (in practice, multiple payment modifiers on same line are rare)
    fn calculate_modifier_adjustment(&self, modifiers: &[String]) -> Result<(Decimal, Option<String>)> {
        if modifiers.is_empty() {
            return Ok((Decimal::ONE, None));
        }

        // Find first payment-affecting modifier
        for modifier in modifiers {
            if let Some(adjustment) = ModifierAdjustment::for_modifier(modifier) {
                return Ok((adjustment.adjustment_percent, Some(adjustment.description)));
            }
        }

        // No payment-affecting modifiers found
        Ok((Decimal::ONE, None))
    }

    /// Calculate payment for professional component only (modifier 26)
    pub fn calculate_professional_component(
        &self,
        hcpcs_code: &str,
        year: i32,
        locality_code: &str,
        pos_code: &str,
        units: Decimal,
    ) -> Result<PaymentCalculation> {
        let modifiers = vec!["26".to_string()];

        // Get base calculation
        let mut calc = self.calculate(hcpcs_code, year, locality_code, pos_code, modifiers.clone(), units)?;

        // For professional component, only work RVU is used
        // PE and MP are for technical component
        let adjusted_work_rvu = calc.work_rvu * calc.work_gpci;
        let cf = ConversionFactor::for_year(year)
            .ok_or_else(|| Error::Validation(format!("Conversion factor not available for year {}", year)))?;

        let base_payment = adjusted_work_rvu * cf.factor;
        let final_payment = base_payment; // Modifier 26 is 100% but only of work component
        let total_payment = final_payment * units;

        calc.adjusted_total_rvu = adjusted_work_rvu;
        calc.base_payment = base_payment;
        calc.final_payment = final_payment;
        calc.total_payment = total_payment;
        calc.modifier_adjustment_percent = Decimal::ONE;
        calc.modifier_adjustment_description = Some("Professional component only (Work RVU)".to_string());

        Ok(calc)
    }

    /// Calculate payment for technical component only (modifier TC)
    pub fn calculate_technical_component(
        &self,
        hcpcs_code: &str,
        year: i32,
        locality_code: &str,
        pos_code: &str,
        units: Decimal,
    ) -> Result<PaymentCalculation> {
        let modifiers = vec!["TC".to_string()];

        // Get base calculation
        let mut calc = self.calculate(hcpcs_code, year, locality_code, pos_code, modifiers.clone(), units)?;

        // For technical component, only PE and MP RVUs are used
        let adjusted_pe_rvu = calc.pe_rvu * calc.pe_gpci;
        let adjusted_mp_rvu = calc.mp_rvu * calc.mp_gpci;
        let adjusted_total_rvu = adjusted_pe_rvu + adjusted_mp_rvu;

        let cf = ConversionFactor::for_year(year)
            .ok_or_else(|| Error::Validation(format!("Conversion factor not available for year {}", year)))?;

        let base_payment = adjusted_total_rvu * cf.factor;
        let final_payment = base_payment;
        let total_payment = final_payment * units;

        calc.adjusted_total_rvu = adjusted_total_rvu;
        calc.base_payment = base_payment;
        calc.final_payment = final_payment;
        calc.total_payment = total_payment;
        calc.modifier_adjustment_percent = Decimal::ONE;
        calc.modifier_adjustment_description = Some("Technical component only (PE + MP RVU)".to_string());

        Ok(calc)
    }

    /// Get RVU lookup reference
    pub fn rvu_lookup(&self) -> &RvuLookup {
        &self.rvu_lookup
    }

    /// Get GPCI lookup reference
    pub fn gpci_lookup(&self) -> &GpciLookup {
        &self.gpci_lookup
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_payment_calculation() {
        let calc = PaymentCalculator::with_sample_data();

        // Calculate 99213 in national average locality, office setting
        let result = calc.calculate(
            "99213",
            2024,
            "99", // National average
            "11", // Office
            vec![],
            Decimal::ONE,
        );

        assert!(result.is_ok());
        let payment = result.unwrap();

        assert_eq!(payment.hcpcs_code, "99213");
        assert_eq!(payment.year, 2024);
        assert!(!payment.is_facility);
        assert_eq!(payment.work_rvu, Decimal::new(110, 2));
        assert_eq!(payment.total_rvu, Decimal::new(230, 2));

        // National average GPCI should be 1.00
        assert_eq!(payment.work_gpci, Decimal::ONE);
        assert_eq!(payment.pe_gpci, Decimal::ONE);
        assert_eq!(payment.mp_gpci, Decimal::ONE);

        // With national GPCI of 1.00, adjusted RVU = unadjusted RVU
        assert_eq!(payment.adjusted_total_rvu, payment.total_rvu);

        // Base payment = Total RVU × CF
        // 2.30 × 33.2875 = 76.56125
        let expected_base = Decimal::new(230, 2) * ConversionFactor::cf_2024().factor;
        assert_eq!(payment.base_payment, expected_base);

        // No modifiers, so final payment = base payment
        assert_eq!(payment.final_payment, payment.base_payment);
        assert_eq!(payment.total_payment, payment.final_payment);
    }

    #[test]
    fn test_payment_with_gpci_adjustment() {
        let calc = PaymentCalculator::with_sample_data();

        // Calculate 99213 in Manhattan (high GPCI)
        let result = calc.calculate(
            "99213",
            2024,
            "01", // Manhattan
            "11", // Office
            vec![],
            Decimal::ONE,
        );

        assert!(result.is_ok());
        let payment = result.unwrap();

        // Manhattan has GPCI > 1.00
        assert!(payment.work_gpci > Decimal::ONE);
        assert!(payment.pe_gpci > Decimal::ONE);
        assert!(payment.mp_gpci > Decimal::ONE);

        // Adjusted RVU should be higher than base RVU
        assert!(payment.adjusted_total_rvu > payment.total_rvu);

        // Payment should be higher in Manhattan than national average
        assert!(payment.base_payment > Decimal::new(7656, 3)); // > $76.56
    }

    #[test]
    fn test_payment_with_bilateral_modifier() {
        let calc = PaymentCalculator::with_sample_data();

        // Calculate with bilateral modifier (150%)
        let result = calc.calculate(
            "99213",
            2024,
            "99",
            "11",
            vec!["50".to_string()],
            Decimal::ONE,
        );

        assert!(result.is_ok());
        let payment = result.unwrap();

        assert_eq!(payment.modifiers, vec!["50".to_string()]);
        assert_eq!(payment.modifier_adjustment_percent, Decimal::new(150, 2)); // 1.50
        assert!(payment.modifier_adjustment_description.is_some());

        // Final payment should be 150% of base
        let expected_final = payment.base_payment * Decimal::new(150, 2);
        assert_eq!(payment.final_payment, expected_final);
    }

    #[test]
    fn test_payment_with_multiple_procedure_modifier() {
        let calc = PaymentCalculator::with_sample_data();

        // Calculate with modifier 51 (50% payment)
        let result = calc.calculate(
            "99213",
            2024,
            "99",
            "11",
            vec!["51".to_string()],
            Decimal::ONE,
        );

        assert!(result.is_ok());
        let payment = result.unwrap();

        assert_eq!(payment.modifier_adjustment_percent, Decimal::new(50, 2)); // 0.50

        // Final payment should be 50% of base
        let expected_final = payment.base_payment * Decimal::new(50, 2);
        assert_eq!(payment.final_payment, expected_final);
    }

    #[test]
    fn test_payment_with_units() {
        let calc = PaymentCalculator::with_sample_data();

        // Calculate with 5 units
        let result = calc.calculate(
            "99213",
            2024,
            "99",
            "11",
            vec![],
            Decimal::new(5, 0),
        );

        assert!(result.is_ok());
        let payment = result.unwrap();

        assert_eq!(payment.units, Decimal::new(5, 0));

        // Total payment should be 5x final payment
        let expected_total = payment.final_payment * Decimal::new(5, 0);
        assert_eq!(payment.total_payment, expected_total);
    }

    #[test]
    fn test_facility_vs_non_facility() {
        let calc = PaymentCalculator::with_sample_data();

        // Calculate in office (non-facility)
        let office_result = calc.calculate(
            "99213",
            2024,
            "99",
            "11", // Office
            vec![],
            Decimal::ONE,
        );

        // Calculate in hospital (facility)
        let hospital_result = calc.calculate(
            "99213",
            2024,
            "99",
            "21", // Inpatient hospital
            vec![],
            Decimal::ONE,
        );

        assert!(office_result.is_ok());
        assert!(hospital_result.is_ok());

        let office_payment = office_result.unwrap();
        let hospital_payment = hospital_result.unwrap();

        assert!(!office_payment.is_facility);
        assert!(hospital_payment.is_facility);

        // PE RVU should be lower for facility
        assert!(hospital_payment.pe_rvu < office_payment.pe_rvu);

        // Office payment should be higher
        assert!(office_payment.base_payment > hospital_payment.base_payment);
    }

    #[test]
    fn test_professional_component() {
        let calc = PaymentCalculator::with_sample_data();

        let result = calc.calculate_professional_component(
            "99213",
            2024,
            "99",
            "11",
            Decimal::ONE,
        );

        assert!(result.is_ok());
        let payment = result.unwrap();

        assert!(payment.modifier_adjustment_description.is_some());
        assert!(payment.modifier_adjustment_description.unwrap().contains("Professional component"));

        // Payment should be less than full global service
        let global_result = calc.calculate("99213", 2024, "99", "11", vec![], Decimal::ONE);
        assert!(global_result.is_ok());
        let global_payment = global_result.unwrap();

        assert!(payment.total_payment < global_payment.total_payment);
    }

    #[test]
    fn test_technical_component() {
        let calc = PaymentCalculator::with_sample_data();

        let result = calc.calculate_technical_component(
            "99213",
            2024,
            "99",
            "11",
            Decimal::ONE,
        );

        assert!(result.is_ok());
        let payment = result.unwrap();

        assert!(payment.modifier_adjustment_description.is_some());
        assert!(payment.modifier_adjustment_description.unwrap().contains("Technical component"));
    }

    #[test]
    fn test_format_payment() {
        let calc = PaymentCalculator::with_sample_data();

        let result = calc.calculate(
            "99213",
            2024,
            "99",
            "11",
            vec![],
            Decimal::ONE,
        );

        assert!(result.is_ok());
        let payment = result.unwrap();

        let formatted = payment.format_payment();
        assert!(formatted.starts_with('$'));
        assert!(formatted.contains('.'));
    }

    #[test]
    fn test_invalid_hcpcs_code() {
        let calc = PaymentCalculator::with_sample_data();

        let result = calc.calculate(
            "INVALID",
            2024,
            "99",
            "11",
            vec![],
            Decimal::ONE,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_locality() {
        let calc = PaymentCalculator::with_sample_data();

        let result = calc.calculate(
            "99213",
            2024,
            "999", // Invalid locality
            "11",
            vec![],
            Decimal::ONE,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_year() {
        let calc = PaymentCalculator::with_sample_data();

        let result = calc.calculate(
            "99213",
            2020, // Year not in sample data
            "99",
            "11",
            vec![],
            Decimal::ONE,
        );

        assert!(result.is_err());
    }
}
