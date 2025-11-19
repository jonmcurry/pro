-- Migration: 054_create_specialty_table
-- Description: Create claims.specialty table and link it to provider_taxonomy
-- Date: 2025-11-18

-- Create specialty lookup table
-- Maps Medicare Specialty Codes to Provider/Supplier Type Descriptions
-- Source: CMS Medicare Provider and Supplier Taxonomy Crosswalk (December 2024)
CREATE TABLE claims.specialty (
    specialty_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    specialty_code VARCHAR(10) NOT NULL UNIQUE,
    specialty_description TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    effective_date DATE DEFAULT '2024-12-10',
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_specialty_code ON claims.specialty(specialty_code);
CREATE INDEX idx_specialty_active ON claims.specialty(is_active) WHERE is_active = true;
CREATE INDEX idx_specialty_description ON claims.specialty USING gin(to_tsvector('english', specialty_description));

COMMENT ON TABLE claims.specialty IS 'Medicare Specialty Codes and Provider/Supplier Type Descriptions from CMS';
COMMENT ON COLUMN claims.specialty.specialty_code IS 'Medicare Specialty Code (e.g., 01, 02, C5)';
COMMENT ON COLUMN claims.specialty.specialty_description IS 'Medicare Provider/Supplier Type Description';

-- Add specialty_id foreign key to claims.provider_taxonomy
-- This links NUCC taxonomy codes to Medicare specialty codes
ALTER TABLE claims.provider_taxonomy
ADD COLUMN specialty_id BIGINT REFERENCES claims.specialty(specialty_id) ON DELETE SET NULL;

CREATE INDEX idx_provider_taxonomy_specialty ON claims.provider_taxonomy(specialty_id);

COMMENT ON COLUMN claims.provider_taxonomy.specialty_id IS 'Links NUCC taxonomy code to Medicare specialty code';

-- Populate claims.specialty with CMS Medicare Specialty Codes
-- Source: CMS Medicare Provider and Supplier Taxonomy Crosswalk (December 10, 2024)
INSERT INTO claims.specialty (specialty_code, specialty_description, notes) VALUES
('01', 'Physician/General Practice', 'Medicare Specialty Code 01'),
('02', 'Physician/General Surgery', 'Medicare Specialty Code 02'),
('03', 'Physician/Allergy/Immunology', 'Medicare Specialty Code 03'),
('04', 'Physician/Otolaryngology', 'Medicare Specialty Code 04'),
('05', 'Physician/Anesthesiology', 'Medicare Specialty Code 05'),
('06', 'Physician/Cardiovascular Disease (Cardiology)', 'Medicare Specialty Code 06'),
('07', 'Physician/Dermatology', 'Medicare Specialty Code 07'),
('08', 'Physician/Family Practice', 'Medicare Specialty Code 08'),
('09', 'Physician/Interventional Pain Management', 'Medicare Specialty Code 09'),
('10', 'Physician/Gastroenterology', 'Medicare Specialty Code 10'),
('11', 'Physician/Internal Medicine', 'Medicare Specialty Code 11'),
('12', 'Physician/Osteopathic Manipulative Medicine', 'Medicare Specialty Code 12'),
('13', 'Physician/Neurology', 'Medicare Specialty Code 13'),
('14', 'Physician/Neurosurgery', 'Medicare Specialty Code 14'),
('15', 'Speech Language Pathologist', 'Medicare Specialty Code 15'),
('16', 'Physician/Obstetrics & Gynecology', 'Medicare Specialty Code 16'),
('17', 'Physician/Hospice and Palliative Care', 'Medicare Specialty Code 17'),
('18', 'Physician/Ophthalmology', 'Medicare Specialty Code 18'),
('19', 'Oral Surgery (Dentist only)', 'Medicare Specialty Code 19'),
('20', 'Physician/Orthopedic Surgery', 'Medicare Specialty Code 20'),
('21', 'Clinical Cardiac Electrophysiology', 'Medicare Specialty Code 21'),
('22', 'Physician/Pathology', 'Medicare Specialty Code 22'),
('23', 'Physician/Sports Medicine', 'Medicare Specialty Code 23'),
('24', 'Physician/Plastic and Reconstructive Surgery', 'Medicare Specialty Code 24'),
('25', 'Physician/Physical Medicine and Rehabilitation', 'Medicare Specialty Code 25'),
('26', 'Physician/Psychiatry', 'Medicare Specialty Code 26'),
('27', 'Physician/Geriatric Psychiatry', 'Medicare Specialty Code 27'),
('28', 'Physician/Colorectal Surgery (Proctology)', 'Medicare Specialty Code 28'),
('29', 'Physician/Pulmonary Disease', 'Medicare Specialty Code 29'),
('30', 'Physician/Diagnostic Radiology', 'Medicare Specialty Code 30'),
('31', 'Intensive Cardiac Rehabilitation', 'Medicare Specialty Code 31'),
('32', 'Anesthesiology Assistant', 'Medicare Specialty Code 32'),
('33', 'Physician/Thoracic Surgery', 'Medicare Specialty Code 33'),
('34', 'Physician/Urology', 'Medicare Specialty Code 34'),
('35', 'Chiropractic', 'Medicare Specialty Code 35'),
('36', 'Physician/Nuclear Medicine', 'Medicare Specialty Code 36'),
('37', 'Physician/Pediatric Medicine', 'Medicare Specialty Code 37'),
('38', 'Physician/Geriatric Medicine', 'Medicare Specialty Code 38'),
('39', 'Physician/Nephrology', 'Medicare Specialty Code 39'),
('40', 'Physician/Hand Surgery', 'Medicare Specialty Code 40'),
('41', 'Optometry', 'Medicare Specialty Code 41'),
('42', 'Certified Nurse Midwife', 'Medicare Specialty Code 42'),
('43', 'Certified Registered Nurse Anesthetist (CRNA)', 'Medicare Specialty Code 43'),
('44', 'Physician/Infectious Disease', 'Medicare Specialty Code 44'),
('45', 'Mammography Center', 'Medicare Specialty Code 45'),
('46', 'Physician/Endocrinology', 'Medicare Specialty Code 46'),
('47', 'Independent Diagnostic Testing Facility (IDTF)', 'Medicare Specialty Code 47'),
('48', 'Podiatry', 'Medicare Specialty Code 48'),
('49', 'Ambulatory Surgical Center', 'Medicare Specialty Code 49'),
('50', 'Nurse Practitioner', 'Medicare Specialty Code 50'),
('51', 'Medical Supply Company with Orthotist', 'Medicare Specialty Code 51'),
('52', 'Medical Supply Company with Prosthetist', 'Medicare Specialty Code 52'),
('53', 'Medical Supply Company with Orthotist-Prosthetist', 'Medicare Specialty Code 53'),
('54', 'Other Medical Supply Company', 'Medicare Specialty Code 54'),
('55', 'Individual Certified Orthotist', 'Medicare Specialty Code 55'),
('56', 'Individual Certified Prosthetist', 'Medicare Specialty Code 56'),
('57', 'Individual Certified Prosthetist-Orthotist', 'Medicare Specialty Code 57'),
('58', 'Medical Supply Company with Pharmacist', 'Medicare Specialty Code 58'),
('59', 'Ambulance Service Provider', 'Medicare Specialty Code 59'),
('60', 'Public Health or Welfare Agency', 'Medicare Specialty Code 60'),
('61', 'Voluntary Health or Charitable Agency', 'Medicare Specialty Code 61'),
('62', 'Psychologist, Clinical', 'Medicare Specialty Code 62'),
('63', 'Portable X-Ray Supplier', 'Medicare Specialty Code 63'),
('64', 'Audiologist', 'Medicare Specialty Code 64'),
('65', 'Physical Therapist in Private Practice', 'Medicare Specialty Code 65'),
('66', 'Physician/Rheumatology', 'Medicare Specialty Code 66'),
('67', 'Occupational Therapist in Private Practice', 'Medicare Specialty Code 67'),
('68', 'Psychologist, Clinical', 'Medicare Specialty Code 68'),
('69', 'Clinical Laboratory', 'Medicare Specialty Code 69'),
('70', 'Clinic or Group Practice', 'Medicare Specialty Code 70'),
('71', 'Registered Dietitian or Nutrition Professional', 'Medicare Specialty Code 71'),
('72', 'Physician/Pain Management', 'Medicare Specialty Code 72'),
('73', 'Mass Immunizer Roster Biller', 'Medicare Specialty Code 73'),
('74', 'Radiation Therapy Center', 'Medicare Specialty Code 74'),
('75', 'Slide Preparation Facility', 'Medicare Specialty Code 75'),
('76', 'Physician/Peripheral Vascular Disease', 'Medicare Specialty Code 76'),
('77', 'Physician/Vascular Surgery', 'Medicare Specialty Code 77'),
('78', 'Physician/Cardiac Surgery', 'Medicare Specialty Code 78'),
('79', 'Physician/Addiction Medicine', 'Medicare Specialty Code 79'),
('80', 'Licensed Clinical Social Worker', 'Medicare Specialty Code 80'),
('81', 'Physician/Critical Care (Intensivists)', 'Medicare Specialty Code 81'),
('82', 'Physician/Hematology', 'Medicare Specialty Code 82'),
('83', 'Physician/Hematology-Oncology', 'Medicare Specialty Code 83'),
('84', 'Physician/Preventive Medicine', 'Medicare Specialty Code 84'),
('85', 'Physician/Maxillofacial Surgery', 'Medicare Specialty Code 85'),
('86', 'Physician/Neuropsychiatry', 'Medicare Specialty Code 86'),
('87', 'All Other Suppliers', 'Medicare Specialty Code 87'),
('88', 'Unknown Supplier/Provider Specialty', 'Medicare Specialty Code 88'),
('89', 'Certified Clinical Nurse Specialist', 'Medicare Specialty Code 89'),
('90', 'Physician/Medical Oncology', 'Medicare Specialty Code 90'),
('91', 'Physician/Surgical Oncology', 'Medicare Specialty Code 91'),
('92', 'Physician/Radiation Oncology', 'Medicare Specialty Code 92'),
('93', 'Physician/Emergency Medicine', 'Medicare Specialty Code 93'),
('94', 'Physician/Interventional Radiology', 'Medicare Specialty Code 94'),
('95', 'Advance Diagnostic Imaging', 'Medicare Specialty Code 95'),
('96', 'Optician', 'Medicare Specialty Code 96'),
('97', 'Physician Assistant', 'Medicare Specialty Code 97'),
('98', 'Physician/Gynecological Oncology', 'Medicare Specialty Code 98'),
('99', 'Physician/Undefined Physician type', 'Medicare Specialty Code 99'),
('A0', 'Hospital-General', 'Medicare Specialty Code A0'),
('A1', 'Skilled Nursing Facility', 'Medicare Specialty Code A1'),
('A2', 'Intermediate Care Nursing Facility', 'Medicare Specialty Code A2'),
('A3', 'Other Nursing Facility', 'Medicare Specialty Code A3'),
('A4', 'Home Health Agency', 'Medicare Specialty Code A4'),
('A5', 'Pharmacy', 'Medicare Specialty Code A5'),
('A6', 'Medical Supply Company with Respiratory Therapist', 'Medicare Specialty Code A6'),
('A7', 'Department Store', 'Medicare Specialty Code A7'),
('A8', 'Grocery Store', 'Medicare Specialty Code A8'),
('A9', 'Indian Health Service facility', 'Medicare Specialty Code A9'),
('B1', 'Oxygen supplier', 'Medicare Specialty Code B1'),
('B2', 'Pedorthic personnel', 'Medicare Specialty Code B2'),
('B3', 'Medical supply company with pedorthic personnel', 'Medicare Specialty Code B3'),
('B4', 'Rehabilitation Agency', 'Medicare Specialty Code B4'),
('B5', 'Ocularist', 'Medicare Specialty Code B5'),
('C0', 'Physician/Sleep Medicine', 'Medicare Specialty Code C0'),
('C3', 'Physician/Interventional Cardiology', 'Medicare Specialty Code C3'),
('C5', 'Dentist', 'Medicare Specialty Code C5'),
('C6', 'Physician/Hospitalist', 'Medicare Specialty Code C6'),
('C7', 'Physician/Advanced Heart Failure and Transplant Cardiology', 'Medicare Specialty Code C7'),
('C8', 'Physician/Medical Toxicology', 'Medicare Specialty Code C8'),
('C9', 'Hematopoietic Cell Transplantation and Cellular Therapy', 'Medicare Specialty Code C9'),
('D1', 'Medicare Diabetes Preventive Program', 'Medicare Specialty Code D1'),
('D3', 'Medical Genetics and Genomics', 'Medicare Specialty Code D3'),
('D4', 'Undersea and Hyperbaric Medicine', 'Medicare Specialty Code D4'),
('D5', 'Opioid Treatment Program', 'Medicare Specialty Code D5'),
('D6', 'Home Infusion Therapy Services', 'Medicare Specialty Code D6'),
('D7', 'Micrographic Dermatologic Surgery', 'Medicare Specialty Code D7'),
('D8', 'Adult Congenital Heart Disease', 'Medicare Specialty Code D8'),
('E1', 'Marriage and Family Therapist', 'Medicare Specialty Code E1'),
('E2', 'Mental Health Counselor', 'Medicare Specialty Code E2')
ON CONFLICT (specialty_code) DO NOTHING;

-- Add trigger for updated_at
CREATE TRIGGER update_specialty_updated_at BEFORE UPDATE ON claims.specialty
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Add comment to claims.provider.specialty indicating it's deprecated in favor of the taxonomy relationship
COMMENT ON COLUMN claims.provider.specialty IS
'DEPRECATED: Use taxonomy_code join to provider_taxonomy.specialty_display or specialty_id join to specialty.specialty_description instead. This column is maintained for backward compatibility but should not be used for new features.';

-- Create a helpful view that shows the provider specialty relationships
CREATE OR REPLACE VIEW claims.provider_specialty_view AS
SELECT
    p.provider_id,
    p.npi,
    p.last_name,
    p.first_name,
    p.taxonomy_code,
    pt.specialty_display AS taxonomy_specialty,
    s.specialty_code AS medicare_specialty_code,
    s.specialty_description AS medicare_specialty_description,
    p.specialty AS legacy_specialty_text
FROM claims.provider p
LEFT JOIN claims.provider_taxonomy pt ON p.taxonomy_code = pt.taxonomy_code
LEFT JOIN claims.specialty s ON pt.specialty_id = s.specialty_id
WHERE p.is_active = true;

COMMENT ON VIEW claims.provider_specialty_view IS
'Convenient view showing provider specialty information from both NUCC taxonomy and Medicare specialty codes. Use this view instead of the deprecated specialty column.';
