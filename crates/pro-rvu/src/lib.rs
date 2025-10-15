// RVU-based reimbursement calculation for Professional SMART
//
// This crate provides Medicare Physician Fee Schedule (MPFS) payment calculation
// using Relative Value Units (RVUs), Geographic Practice Cost Indices (GPCIs),
// conversion factors, and modifier adjustments.
//
// # Payment Formula
//
// ```text
// Payment = [(Work RVU × Work GPCI) + (PE RVU × PE GPCI) + (MP RVU × MP GPCI)] × CF × Modifier % × Units
// ```
//
// Where:
// - Work RVU: Relative value for physician work
// - PE RVU: Practice expense (facility or non-facility)
// - MP RVU: Malpractice expense
// - GPCI: Geographic Practice Cost Index (locality adjustment)
// - CF: Conversion Factor (dollar amount per RVU)
// - Modifier %: Payment adjustment based on modifiers (50, 51, 52, 26, TC, etc.)
// - Units: Number of units billed

pub mod calculator;
pub mod lookup;
pub mod types;

// Re-export commonly used items
pub use calculator::{PaymentCalculation, PaymentCalculator};
pub use lookup::{GpciLookup, RvuLookup};
pub use types::{
    ConversionFactor, GpciData, ModifierAdjustment, PlaceOfService, RvuData,
};
pub use pro_common::{Error, Result};
