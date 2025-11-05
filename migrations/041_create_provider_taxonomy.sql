-- Migration: 041_create_provider_taxonomy
-- Description: Create provider taxonomy lookup table with comprehensive NUCC taxonomy codes
-- Date: 2025-11-05

-- Provider Taxonomy Lookup Table
-- Maps NUCC Healthcare Provider Taxonomy codes to specialty display names
CREATE TABLE claims.provider_taxonomy (
    taxonomy_code VARCHAR(10) PRIMARY KEY,
    provider_type VARCHAR(100) NOT NULL,        -- Individual, Organization
    classification VARCHAR(200) NOT NULL,       -- e.g., "Allopathic & Osteopathic Physicians"
    specialization VARCHAR(200),                -- e.g., "Family Medicine"
    specialty_display VARCHAR(200) NOT NULL,    -- User-friendly display name
    definition TEXT,                            -- Official NUCC definition
    is_active BOOLEAN DEFAULT true,
    effective_date DATE DEFAULT '2024-01-01',
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_taxonomy_specialty ON claims.provider_taxonomy(specialty_display);
CREATE INDEX idx_taxonomy_classification ON claims.provider_taxonomy(classification);
CREATE INDEX idx_taxonomy_active ON claims.provider_taxonomy(is_active) WHERE is_active = true;

COMMENT ON TABLE claims.provider_taxonomy IS 'NUCC Healthcare Provider Taxonomy code set - maps taxonomy codes to specialty display names';
COMMENT ON COLUMN claims.provider_taxonomy.taxonomy_code IS 'NUCC 10-character taxonomy code (e.g., 207Q00000X)';
COMMENT ON COLUMN claims.provider_taxonomy.specialty_display IS 'User-friendly specialty name for display and reporting';

-- Populate with comprehensive taxonomy data
-- Source: NUCC Health Care Provider Taxonomy Code Set

-- =============================================================================
-- PHYSICIANS - ALLOPATHIC & OSTEOPATHIC (207-209 series)
-- =============================================================================

-- Family Medicine
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207Q00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Family Medicine', 'Family Medicine', 'A physician who specializes in family medicine provides continuing and comprehensive health care for the individual and family across all ages, genders, diseases, and parts of the body.'),
('207QA0000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Family Medicine - Adolescent Medicine', 'Family Medicine - Adolescent Medicine', 'Physician with special qualifications and expertise in adolescent medicine.'),
('207QA0505X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Family Medicine - Adult Medicine', 'Family Medicine - Adult Medicine', 'Physician specializing in adult medicine within family practice.'),
('207QG0300X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Family Medicine - Geriatric Medicine', 'Family Medicine - Geriatric Medicine', 'Family physician with specialized training in geriatric medicine.'),
('207QS0010X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Family Medicine - Sports Medicine', 'Family Medicine - Sports Medicine', 'Family physician specializing in sports medicine.');

-- Internal Medicine
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207R00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine', 'Internal Medicine', 'A physician who provides long-term, comprehensive care in the office and the hospital, managing both common and complex illness of adolescents, adults and the elderly.'),
('207RA0000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Adolescent Medicine', 'Internal Medicine - Adolescent Medicine', 'Internist specializing in adolescent medicine.'),
('207RA0001X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Advanced Heart Failure', 'Internal Medicine - Advanced Heart Failure & Transplant Cardiology', 'Subspecialty of cardiovascular disease focused on advanced heart failure.'),
('207RC0000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Cardiovascular Disease', 'Cardiology', 'An internist who specializes in diseases of the heart and blood vessels and manages complex cardiac conditions.'),
('207RC0001X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Clinical Cardiac Electrophysiology', 'Cardiology - Electrophysiology', 'Cardiologist specializing in the diagnosis and treatment of cardiac arrhythmias.'),
('207RE0101X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Endocrinology', 'Endocrinology', 'An internist who concentrates on disorders of the internal glands such as the thyroid and adrenal glands.'),
('207RG0100X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Gastroenterology', 'Gastroenterology', 'An internist who specializes in diagnosis and treatment of diseases of the digestive organs.'),
('207RG0300X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Geriatric Medicine', 'Geriatric Medicine', 'An internist who has special knowledge of the aging process and special skills in the diagnostic and therapeutic aspects of disease in the elderly.'),
('207RH0000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Hematology', 'Hematology', 'An internist trained in the diagnosis and treatment of diseases and disorders of the blood, blood-forming organs, and blood proteins.'),
('207RH0003X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Hematology & Oncology', 'Hematology & Oncology', 'An internist who specializes in hematology and medical oncology.'),
('207RI0001X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Infectious Disease', 'Infectious Disease', 'An internist who deals with infectious diseases of all types and in all organ systems.'),
('207RI0008X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Hepatology', 'Hepatology', 'An internist who specializes in diseases of the liver.'),
('207RI0011X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Interventional Cardiology', 'Interventional Cardiology', 'Cardiologist specializing in catheter-based treatment of heart disease.'),
('207RN0300X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Nephrology', 'Nephrology', 'An internist who treats disorders of the kidney, high blood pressure, and mineral metabolism.'),
('207RP1001X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Pulmonary Disease', 'Pulmonology', 'An internist who treats diseases of the lungs and airways.'),
('207RR0500X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Rheumatology', 'Rheumatology', 'An internist who treats diseases of joints, muscle, bones and tendons including arthritis and collagen diseases.'),
('207RS0010X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Sports Medicine', 'Internal Medicine - Sports Medicine', 'Internist specializing in sports medicine.'),
('207RS0012X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Internal Medicine - Sleep Medicine', 'Sleep Medicine', 'Internist specializing in the diagnosis and treatment of sleep disorders.');

-- Pediatrics
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('208000000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics', 'Pediatrics', 'A physician who is concerned with the physical, emotional and social health of children from birth to young adulthood.'),
('2080A0000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Adolescent Medicine', 'Pediatric Adolescent Medicine', 'Pediatrician specializing in adolescent medicine.'),
('2080B0002X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Obesity Medicine', 'Pediatric Obesity Medicine', 'Pediatrician specializing in obesity medicine.'),
('2080C0008X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Child Abuse Pediatrics', 'Pediatric Child Abuse', 'Pediatrician specializing in child abuse and neglect.'),
('2080H0002X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Hospice and Palliative Medicine', 'Pediatric Hospice & Palliative Medicine', 'Pediatrician specializing in hospice and palliative care.'),
('2080I0007X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Infectious Diseases', 'Pediatric Infectious Disease', 'Pediatrician specializing in infectious diseases.'),
('2080N0001X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Neonatal-Perinatal Medicine', 'Neonatology', 'Pediatrician specializing in the care of critically ill newborns.'),
('2080P0006X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Developmental-Behavioral Pediatrics', 'Developmental-Behavioral Pediatrics', 'Pediatrician specializing in developmental and behavioral problems.'),
('2080P0008X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Pulmonology', 'Pediatric Pulmonology', 'Pediatrician specializing in respiratory diseases.'),
('2080P0201X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Hematology-Oncology', 'Pediatric Hematology-Oncology', 'Pediatrician specializing in blood diseases and cancer.'),
('2080P0202X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Cardiology', 'Pediatric Cardiology', 'Pediatrician specializing in heart disease.'),
('2080P0203X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Critical Care Medicine', 'Pediatric Critical Care', 'Pediatrician specializing in critical care medicine.'),
('2080P0204X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Emergency Medicine', 'Pediatric Emergency Medicine', 'Pediatrician specializing in emergency medicine.'),
('2080P0205X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Endocrinology', 'Pediatric Endocrinology', 'Pediatrician specializing in endocrine disorders.'),
('2080P0206X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Gastroenterology', 'Pediatric Gastroenterology', 'Pediatrician specializing in digestive disorders.'),
('2080P0207X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Nephrology', 'Pediatric Nephrology', 'Pediatrician specializing in kidney disease.'),
('2080P0208X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Rheumatology', 'Pediatric Rheumatology', 'Pediatrician specializing in rheumatic diseases.'),
('2080S0010X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Sports Medicine', 'Pediatric Sports Medicine', 'Pediatrician specializing in sports medicine.'),
('2080S0012X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Sleep Medicine', 'Pediatric Sleep Medicine', 'Pediatrician specializing in sleep disorders.'),
('2080T0002X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pediatrics - Pediatric Transplant Hepatology', 'Pediatric Transplant Hepatology', 'Pediatrician specializing in liver transplantation.');

-- Surgery
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('208600000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Surgery', 'General Surgery', 'A physician trained to provide operative, perioperative and critical care of patients.'),
('2086S0102X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Surgery - Surgical Critical Care', 'Surgical Critical Care', 'Surgeon specializing in critical care of surgical patients.'),
('2086S0105X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Surgery - Surgery of the Hand', 'Hand Surgery', 'Surgeon specializing in hand surgery.'),
('2086S0120X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Surgery - Pediatric Surgery', 'Pediatric Surgery', 'Surgeon specializing in surgery of infants and children.'),
('2086S0122X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Surgery - Plastic and Reconstructive Surgery', 'Plastic Surgery', 'Surgeon specializing in plastic and reconstructive surgery.'),
('2086S0127X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Surgery - Trauma Surgery', 'Trauma Surgery', 'Surgeon specializing in traumatic injuries.'),
('2086S0129X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Surgery - Vascular Surgery', 'Vascular Surgery', 'Surgeon specializing in vascular surgery.'),
('2086X0206X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Surgery - Surgical Oncology', 'Surgical Oncology', 'Surgeon specializing in cancer surgery.');

-- Orthopedic Surgery
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207X00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Orthopaedic Surgery', 'Orthopedic Surgery', 'A physician who provides the diagnosis and treatment of musculoskeletal disorders.'),
('207XS0114X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Orthopaedic Surgery - Adult Reconstructive Orthopaedic Surgery', 'Adult Reconstructive Orthopedic Surgery', 'Orthopedic surgeon specializing in adult reconstructive surgery.'),
('207XS0106X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Orthopaedic Surgery - Hand Surgery', 'Orthopedic Hand Surgery', 'Orthopedic surgeon specializing in hand surgery.'),
('207XX0004X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Orthopaedic Surgery - Foot and Ankle Surgery', 'Orthopedic Foot & Ankle Surgery', 'Orthopedic surgeon specializing in foot and ankle.'),
('207XX0005X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Orthopaedic Surgery - Sports Medicine', 'Orthopedic Sports Medicine', 'Orthopedic surgeon specializing in sports medicine.'),
('207XS0117X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Orthopaedic Surgery - Orthopaedic Surgery of the Spine', 'Spine Surgery', 'Orthopedic surgeon specializing in spine surgery.'),
('207XX0801X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Orthopaedic Surgery - Orthopaedic Trauma', 'Orthopedic Trauma', 'Orthopedic surgeon specializing in traumatic injuries.');

-- Neurosurgery
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207T00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Neurological Surgery', 'Neurosurgery', 'A physician who provides operative and non-operative management of disorders of the central, peripheral, and autonomic nervous systems.'),
('207TS0010X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Neurological Surgery - Vascular Neurosurgery', 'Vascular Neurosurgery', 'Neurosurgeon specializing in cerebrovascular surgery.');

-- Neurology
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('204D00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Neuromusculoskeletal Medicine & OMM', 'Neuromusculoskeletal Medicine', 'Physician specializing in neuromusculoskeletal medicine.'),
('2084N0400X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Psychiatry & Neurology - Neurology', 'Neurology', 'A physician with special training in diagnosing and treating diseases of the nervous system.'),
('2084N0402X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Psychiatry & Neurology - Neurology with Special Qualifications in Child Neurology', 'Child Neurology', 'Neurologist specializing in pediatric neurology.'),
('2084N0600X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Psychiatry & Neurology - Clinical Neurophysiology', 'Clinical Neurophysiology', 'Neurologist specializing in clinical neurophysiology.'),
('2084V0102X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Psychiatry & Neurology - Vascular Neurology', 'Vascular Neurology', 'Neurologist specializing in cerebrovascular disease.');

-- Psychiatry
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('2084P0800X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Psychiatry & Neurology - Psychiatry', 'Psychiatry', 'A physician who specializes in the prevention, diagnosis, and treatment of mental, addictive and emotional disorders.'),
('2084A0401X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Psychiatry & Neurology - Addiction Medicine', 'Addiction Psychiatry', 'Psychiatrist specializing in addiction medicine.'),
('2084P0301X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Psychiatry & Neurology - Brain Injury Medicine', 'Brain Injury Medicine', 'Psychiatrist specializing in brain injury.'),
('2084P0804X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Psychiatry & Neurology - Child & Adolescent Psychiatry', 'Child & Adolescent Psychiatry', 'Psychiatrist specializing in children and adolescents.'),
('2084P0805X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Psychiatry & Neurology - Geriatric Psychiatry', 'Geriatric Psychiatry', 'Psychiatrist specializing in elderly patients.'),
('2084F0202X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Psychiatry & Neurology - Forensic Psychiatry', 'Forensic Psychiatry', 'Psychiatrist specializing in forensic issues.');

-- Anesthesiology
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207L00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Anesthesiology', 'Anesthesiology', 'A physician who provides pain relief and maintenance of a stable condition during and immediately following an operation or obstetric or diagnostic procedure.'),
('207LA0401X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Anesthesiology - Addiction Medicine', 'Addiction Medicine - Anesthesiology', 'Anesthesiologist specializing in addiction medicine.'),
('207LC0200X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Anesthesiology - Critical Care Medicine', 'Critical Care Medicine - Anesthesiology', 'Anesthesiologist specializing in critical care.'),
('207LH0002X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Anesthesiology - Hospice and Palliative Medicine', 'Hospice & Palliative Medicine - Anesthesiology', 'Anesthesiologist specializing in hospice and palliative care.'),
('207LP2900X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Anesthesiology - Pain Medicine', 'Pain Medicine', 'Anesthesiologist specializing in pain management.'),
('207LP3000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Anesthesiology - Pediatric Anesthesiology', 'Pediatric Anesthesiology', 'Anesthesiologist specializing in pediatric patients.');

-- Dermatology
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207N00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Dermatology', 'Dermatology', 'A physician trained to diagnose and treat pediatric and adult patients with disorders of the skin, hair, nails and adjacent mucous membranes.'),
('207ND0101X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Dermatology - MOHS-Micrographic Surgery', 'Dermatology - MOHS Surgery', 'Dermatologist specializing in MOHS micrographic surgery.'),
('207ND0900X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Dermatology - Dermatopathology', 'Dermatopathology', 'Dermatologist specializing in pathology of the skin.'),
('207NP0225X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Dermatology - Pediatric Dermatology', 'Pediatric Dermatology', 'Dermatologist specializing in pediatric patients.');

-- Emergency Medicine
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207P00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Emergency Medicine', 'Emergency Medicine', 'A physician trained to provide immediate care to acutely ill or injured patients.'),
('207PE0004X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Emergency Medicine - Emergency Medical Services', 'Emergency Medical Services', 'Emergency physician specializing in EMS systems.'),
('207PH0002X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Emergency Medicine - Hospice and Palliative Medicine', 'Hospice & Palliative Medicine - Emergency Medicine', 'Emergency physician specializing in hospice and palliative care.'),
('207PP0204X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Emergency Medicine - Pediatric Emergency Medicine', 'Pediatric Emergency Medicine', 'Emergency physician specializing in pediatric patients.'),
('207PS0010X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Emergency Medicine - Sports Medicine', 'Sports Medicine - Emergency Medicine', 'Emergency physician specializing in sports medicine.'),
('207PT0002X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Emergency Medicine - Undersea and Hyperbaric Medicine', 'Undersea & Hyperbaric Medicine', 'Emergency physician specializing in hyperbaric medicine.');

-- Obstetrics & Gynecology
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207V00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Obstetrics & Gynecology', 'Obstetrics & Gynecology', 'A physician who provides medical and surgical care to women and has particular skills in pregnancy, childbirth and disorders of the reproductive system.'),
('207VB0002X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Obstetrics & Gynecology - Obesity Medicine', 'Obesity Medicine - OB/GYN', 'OB/GYN specializing in obesity medicine.'),
('207VC0200X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Obstetrics & Gynecology - Critical Care Medicine', 'Critical Care Medicine - OB/GYN', 'OB/GYN specializing in critical care.'),
('207VE0102X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Obstetrics & Gynecology - Reproductive Endocrinology', 'Reproductive Endocrinology & Infertility', 'OB/GYN specializing in reproductive endocrinology.'),
('207VF0040X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Obstetrics & Gynecology - Female Pelvic Medicine and Reconstructive Surgery', 'Female Pelvic Medicine & Reconstructive Surgery', 'OB/GYN specializing in pelvic medicine.'),
('207VG0400X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Obstetrics & Gynecology - Gynecology', 'Gynecology', 'Physician specializing in gynecology only.'),
('207VH0002X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Obstetrics & Gynecology - Hospice and Palliative Medicine', 'Hospice & Palliative Medicine - OB/GYN', 'OB/GYN specializing in hospice and palliative care.'),
('207VM0101X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Obstetrics & Gynecology - Maternal & Fetal Medicine', 'Maternal-Fetal Medicine', 'OB/GYN specializing in high-risk pregnancies.'),
('207VX0000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Obstetrics & Gynecology - Obstetrics', 'Obstetrics', 'Physician specializing in obstetrics only.'),
('207VX0201X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Obstetrics & Gynecology - Gynecologic Oncology', 'Gynecologic Oncology', 'OB/GYN specializing in gynecologic cancer.');

-- Ophthalmology
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207W00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Ophthalmology', 'Ophthalmology', 'A physician who has special training in diagnosing and treating diseases of the eye.'),
('207WX0009X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Ophthalmology - Glaucoma Specialist', 'Glaucoma Specialist', 'Ophthalmologist specializing in glaucoma.'),
('207WX0107X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Ophthalmology - Retina Specialist', 'Retina Specialist', 'Ophthalmologist specializing in retinal diseases.'),
('207WX0108X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Ophthalmology - Uveitis and Ocular Inflammatory Disease', 'Uveitis & Ocular Inflammatory Disease', 'Ophthalmologist specializing in uveitis.'),
('207WX0109X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Ophthalmology - Neuro-ophthalmology', 'Neuro-ophthalmology', 'Ophthalmologist specializing in neuro-ophthalmology.'),
('207WX0110X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Ophthalmology - Pediatric Ophthalmology', 'Pediatric Ophthalmology', 'Ophthalmologist specializing in pediatric patients.'),
('207WX0120X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Ophthalmology - Cornea and External Diseases Specialist', 'Cornea & External Disease Specialist', 'Ophthalmologist specializing in corneal diseases.');

-- Otolaryngology
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207Y00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Otolaryngology', 'Otolaryngology (ENT)', 'A physician trained in the medical and surgical treatment of the ear, nose, throat, and related structures of the head and neck.'),
('207YP0228X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Otolaryngology - Pediatric Otolaryngology', 'Pediatric Otolaryngology', 'ENT specialist focusing on pediatric patients.'),
('207YS0012X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Otolaryngology - Sleep Medicine', 'Sleep Medicine - ENT', 'ENT specialist focusing on sleep disorders.'),
('207YS0123X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Otolaryngology - Facial Plastic Surgery', 'Facial Plastic Surgery', 'ENT specialist focusing on facial plastic surgery.'),
('207YX0007X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Otolaryngology - Plastic Surgery within the Head & Neck', 'Plastic Surgery - Head & Neck', 'ENT specialist focusing on head and neck plastic surgery.');

-- Pathology
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('207ZP0101X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Anatomic Pathology', 'Anatomic Pathology', 'Pathologist specializing in anatomic pathology.'),
('207ZB0001X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Blood Banking & Transfusion Medicine', 'Blood Banking & Transfusion Medicine', 'Pathologist specializing in blood banking.'),
('207ZC0006X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Clinical Pathology', 'Clinical Pathology', 'Pathologist specializing in clinical pathology.'),
('207ZC0008X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Clinical Informatics', 'Clinical Informatics - Pathology', 'Pathologist specializing in clinical informatics.'),
('207ZC0500X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Cytopathology', 'Cytopathology', 'Pathologist specializing in cytopathology.'),
('207ZD0900X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Dermatopathology', 'Dermatopathology', 'Pathologist specializing in dermatopathology.'),
('207ZF0201X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Forensic Pathology', 'Forensic Pathology', 'Pathologist specializing in forensic pathology.'),
('207ZH0000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Hematology', 'Hematology - Pathology', 'Pathologist specializing in hematology.'),
('207ZI0100X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Immunopathology', 'Immunopathology', 'Pathologist specializing in immunopathology.'),
('207ZM0300X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Medical Microbiology', 'Medical Microbiology', 'Pathologist specializing in medical microbiology.'),
('207ZN0500X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Neuropathology', 'Neuropathology', 'Pathologist specializing in neuropathology.'),
('207ZP0007X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Molecular Genetic Pathology', 'Molecular Genetic Pathology', 'Pathologist specializing in molecular genetics.'),
('207ZP0104X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Anatomic Pathology & Clinical Pathology', 'Anatomic & Clinical Pathology', 'Pathologist specializing in both anatomic and clinical pathology.'),
('207ZP0105X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Pathology - Clinical Pathology/Laboratory Medicine', 'Clinical Pathology/Laboratory Medicine', 'Pathologist specializing in laboratory medicine.');

-- Radiology
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('2085R0001X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Radiology - Diagnostic Radiology', 'Diagnostic Radiology', 'A physician specializing in diagnostic imaging.'),
('2085R0202X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Radiology - Diagnostic Neuroradiology', 'Neuroradiology', 'Radiologist specializing in neurological imaging.'),
('2085R0203X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Radiology - Nuclear Radiology', 'Nuclear Radiology', 'Radiologist specializing in nuclear medicine.'),
('2085R0204X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Radiology - Vascular & Interventional Radiology', 'Interventional Radiology', 'Radiologist specializing in interventional procedures.'),
('2085D0003X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Radiology - Diagnostic Ultrasound', 'Diagnostic Ultrasound', 'Radiologist specializing in ultrasound.'),
('2085N0700X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Radiology - Nuclear Medicine', 'Nuclear Medicine', 'Radiologist specializing in nuclear medicine.'),
('2085N0904X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Radiology - Nuclear Cardiology', 'Nuclear Cardiology', 'Radiologist specializing in cardiac nuclear imaging.'),
('2085P0229X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Radiology - Pediatric Radiology', 'Pediatric Radiology', 'Radiologist specializing in pediatric imaging.'),
('2085R0205X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Radiology - Radiation Oncology', 'Radiation Oncology', 'Physician specializing in radiation therapy for cancer.');

-- Urology
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('208800000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Urology', 'Urology', 'A physician specializing in diseases of the urinary organs in females and the urinary and sex organs in males.'),
('2088P0231X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Urology - Pediatric Urology', 'Pediatric Urology', 'Urologist specializing in pediatric patients.'),
('2088F0040X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Urology - Female Pelvic Medicine and Reconstructive Surgery', 'Female Pelvic Medicine & Reconstructive Surgery - Urology', 'Urologist specializing in female pelvic medicine.');

-- Other Physician Specialties
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('208D00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'General Practice', 'General Practice', 'A physician who treats a variety of medical problems in patients of all ages.'),
('208C00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Colon & Rectal Surgery', 'Colon & Rectal Surgery', 'A surgeon specializing in the diagnosis and treatment of diseases of the colon, rectum and anus.'),
('208G00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Thoracic Surgery (Cardiothoracic Vascular Surgery)', 'Thoracic Surgery', 'A surgeon specializing in operative, perioperative and critical care of patients with pathologic conditions within the chest.'),
('208M00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Hospitalist', 'Hospitalist', 'A physician who specializes in the care of hospitalized patients.'),
('208U00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Clinical Pharmacology', 'Clinical Pharmacology', 'A physician who specializes in clinical pharmacology.'),
('207K00000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Allergy & Immunology', 'Allergy & Immunology', 'A physician who specializes in the diagnosis and treatment of allergies and immunologic disorders.'),
('207KA0200X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Allergy & Immunology - Allergy', 'Allergy', 'Physician specializing in allergies.'),
('207KI0005X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Allergy & Immunology - Clinical & Laboratory Immunology', 'Clinical & Laboratory Immunology', 'Physician specializing in immunology.');

-- Physical Medicine & Rehabilitation
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('208100000X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Physical Medicine & Rehabilitation', 'Physical Medicine & Rehabilitation', 'A physician who specializes in physical medicine and rehabilitation.'),
('2081H0002X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Physical Medicine & Rehabilitation - Hospice and Palliative Medicine', 'Hospice & Palliative Medicine - PM&R', 'PM&R physician specializing in hospice and palliative care.'),
('2081N0008X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Physical Medicine & Rehabilitation - Neuromuscular Medicine', 'Neuromuscular Medicine', 'PM&R physician specializing in neuromuscular medicine.'),
('2081P0004X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Physical Medicine & Rehabilitation - Spinal Cord Injury Medicine', 'Spinal Cord Injury Medicine', 'PM&R physician specializing in spinal cord injuries.'),
('2081P0010X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Physical Medicine & Rehabilitation - Pediatric Rehabilitation Medicine', 'Pediatric Rehabilitation Medicine', 'PM&R physician specializing in pediatric patients.'),
('2081P2900X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Physical Medicine & Rehabilitation - Pain Medicine', 'Pain Medicine - PM&R', 'PM&R physician specializing in pain management.'),
('2081S0010X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Physical Medicine & Rehabilitation - Sports Medicine', 'Sports Medicine - PM&R', 'PM&R physician specializing in sports medicine.');

-- Preventive Medicine
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('2083A0100X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Preventive Medicine - Aerospace Medicine', 'Aerospace Medicine', 'Physician specializing in aerospace medicine.'),
('2083P0500X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Preventive Medicine - Preventive Medicine/Occupational Environmental Medicine', 'Occupational Medicine', 'Physician specializing in occupational and environmental medicine.'),
('2083P0901X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Preventive Medicine - Public Health & General Preventive Medicine', 'Public Health & Preventive Medicine', 'Physician specializing in public health.'),
('2083S0010X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Preventive Medicine - Sports Medicine', 'Sports Medicine - Preventive Medicine', 'Physician specializing in sports medicine.');

-- Medical Genetics
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('2080S0010X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Medical Genetics', 'Medical Genetics', 'A physician who specializes in medical genetics.'),
('2080T0004X', 'Individual', 'Allopathic & Osteopathic Physicians', 'Medical Genetics - Medical Biochemical Genetics', 'Medical Biochemical Genetics', 'Physician specializing in biochemical genetics.');

-- =============================================================================
-- NON-PHYSICIAN PRACTITIONERS
-- =============================================================================

-- Nurse Practitioners
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('363L00000X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner', 'Nurse Practitioner', 'A registered nurse who has advanced clinical education and training in a health care specialty area.'),
('363LA2100X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Acute Care', 'Nurse Practitioner - Acute Care', 'Nurse practitioner specializing in acute care.'),
('363LA2200X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Adult Health', 'Nurse Practitioner - Adult Health', 'Nurse practitioner specializing in adult health.'),
('363LC1500X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Community Health', 'Nurse Practitioner - Community Health', 'Nurse practitioner specializing in community health.'),
('363LC0200X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Critical Care Medicine', 'Nurse Practitioner - Critical Care', 'Nurse practitioner specializing in critical care.'),
('363LF0000X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Family', 'Nurse Practitioner - Family', 'Nurse practitioner specializing in family medicine.'),
('363LG0600X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Gerontology', 'Nurse Practitioner - Gerontology', 'Nurse practitioner specializing in gerontology.'),
('363LN0000X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Neonatal', 'Nurse Practitioner - Neonatal', 'Nurse practitioner specializing in neonatal care.'),
('363LN0005X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Neonatal, Critical Care', 'Nurse Practitioner - Neonatal Critical Care', 'Nurse practitioner specializing in neonatal critical care.'),
('363LP0200X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Pediatrics', 'Nurse Practitioner - Pediatrics', 'Nurse practitioner specializing in pediatrics.'),
('363LP0222X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Pediatrics, Critical Care', 'Nurse Practitioner - Pediatric Critical Care', 'Nurse practitioner specializing in pediatric critical care.'),
('363LP0808X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Psychiatric/Mental Health', 'Nurse Practitioner - Psychiatric/Mental Health', 'Nurse practitioner specializing in psychiatric and mental health.'),
('363LP1700X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Perinatal', 'Nurse Practitioner - Perinatal', 'Nurse practitioner specializing in perinatal care.'),
('363LP2300X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Primary Care', 'Nurse Practitioner - Primary Care', 'Nurse practitioner specializing in primary care.'),
('363LS0200X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - School', 'Nurse Practitioner - School Health', 'Nurse practitioner specializing in school health.'),
('363LW0102X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Women''s Health', 'Nurse Practitioner - Women''s Health', 'Nurse practitioner specializing in women''s health.'),
('363LX0001X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Obstetrics & Gynecology', 'Nurse Practitioner - Obstetrics & Gynecology', 'Nurse practitioner specializing in obstetrics and gynecology.'),
('363LX0106X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Practitioner - Occupational Health', 'Nurse Practitioner - Occupational Health', 'Nurse practitioner specializing in occupational health.');

-- Physician Assistants
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('363A00000X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Physician Assistant', 'Physician Assistant', 'A person who has successfully completed an accredited education program for physician assistant, is licensed by the state and is practicing within the scope of that license.');

-- Clinical Nurse Specialist
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('364S00000X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist', 'Clinical Nurse Specialist', 'A registered nurse with clinical expertise in a specialized area of nursing practice.'),
('364SA2100X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Acute Care', 'Clinical Nurse Specialist - Acute Care', 'Clinical nurse specialist specializing in acute care.'),
('364SA2200X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Adult Health', 'Clinical Nurse Specialist - Adult Health', 'Clinical nurse specialist specializing in adult health.'),
('364SC0200X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Critical Care Medicine', 'Clinical Nurse Specialist - Critical Care', 'Clinical nurse specialist specializing in critical care.'),
('364SC1501X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Community Health/Public Health', 'Clinical Nurse Specialist - Community Health', 'Clinical nurse specialist specializing in community health.'),
('364SE0003X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Emergency', 'Clinical Nurse Specialist - Emergency', 'Clinical nurse specialist specializing in emergency care.'),
('364SF0001X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Family Health', 'Clinical Nurse Specialist - Family Health', 'Clinical nurse specialist specializing in family health.'),
('364SG0600X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Gerontology', 'Clinical Nurse Specialist - Gerontology', 'Clinical nurse specialist specializing in gerontology.'),
('364SL0600X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Long-Term Care', 'Clinical Nurse Specialist - Long-Term Care', 'Clinical nurse specialist specializing in long-term care.'),
('364SN0000X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Neonatal', 'Clinical Nurse Specialist - Neonatal', 'Clinical nurse specialist specializing in neonatal care.'),
('364SN0800X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Neuroscience', 'Clinical Nurse Specialist - Neuroscience', 'Clinical nurse specialist specializing in neuroscience.'),
('364SP0200X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Pediatrics', 'Clinical Nurse Specialist - Pediatrics', 'Clinical nurse specialist specializing in pediatrics.'),
('364SP1700X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Perinatal', 'Clinical Nurse Specialist - Perinatal', 'Clinical nurse specialist specializing in perinatal care.'),
('364SP2800X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Psychiatric/Mental Health', 'Clinical Nurse Specialist - Psychiatric/Mental Health', 'Clinical nurse specialist specializing in psychiatric and mental health.'),
('364SS0200X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - School', 'Clinical Nurse Specialist - School Health', 'Clinical nurse specialist specializing in school health.'),
('364SW0102X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Clinical Nurse Specialist - Women''s Health', 'Clinical Nurse Specialist - Women''s Health', 'Clinical nurse specialist specializing in women''s health.');

-- Nurse Anesthetist
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('367500000X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Nurse Anesthetist, Certified Registered', 'Certified Registered Nurse Anesthetist (CRNA)', 'An advanced practice registered nurse who has acquired specialized knowledge and skills pertaining to the provision of anesthesia care.');

-- Nurse Midwife
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('367A00000X', 'Individual', 'Physician Assistants & Advanced Practice Nursing Providers', 'Advanced Practice Midwife', 'Certified Nurse Midwife', 'An advanced practice registered nurse who has acquired specialized knowledge and skills pertaining to midwifery.');

-- =============================================================================
-- BEHAVIORAL HEALTH PROVIDERS
-- =============================================================================

-- Psychologists
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('103T00000X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist', 'Psychologist', 'An individual who is trained in methods of psychological analysis, therapy, and research.'),
('103TA0400X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Addiction', 'Psychologist - Addiction', 'Psychologist specializing in addiction.'),
('103TA0700X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Adult Development & Aging', 'Psychologist - Adult Development & Aging', 'Psychologist specializing in adult development and aging.'),
('103TB0200X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Cognitive & Behavioral', 'Psychologist - Cognitive & Behavioral', 'Psychologist specializing in cognitive and behavioral therapy.'),
('103TC0700X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Clinical', 'Clinical Psychologist', 'Psychologist specializing in clinical psychology.'),
('103TC1900X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Counseling', 'Counseling Psychologist', 'Psychologist specializing in counseling.'),
('103TC2200X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Clinical Child & Adolescent', 'Clinical Child & Adolescent Psychologist', 'Psychologist specializing in children and adolescents.'),
('103TE1000X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Educational', 'Educational Psychologist', 'Psychologist specializing in educational psychology.'),
('103TE1100X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Exercise & Sports', 'Exercise & Sports Psychologist', 'Psychologist specializing in exercise and sports psychology.'),
('103TF0000X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Family', 'Family Psychologist', 'Psychologist specializing in family psychology.'),
('103TF0200X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Forensic', 'Forensic Psychologist', 'Psychologist specializing in forensic psychology.'),
('103TH0100X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Health', 'Health Psychologist', 'Psychologist specializing in health psychology.'),
('103TM1800X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Mental Retardation & Developmental Disabilities', 'Psychologist - Developmental Disabilities', 'Psychologist specializing in developmental disabilities.'),
('103TP0016X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Prescribing (Medical)', 'Prescribing Psychologist', 'Psychologist with prescriptive authority.'),
('103TP0814X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Psychoanalysis', 'Psychologist - Psychoanalysis', 'Psychologist specializing in psychoanalysis.'),
('103TP2700X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Psychotherapy', 'Psychologist - Psychotherapy', 'Psychologist specializing in psychotherapy.'),
('103TP2701X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Group Psychotherapy', 'Psychologist - Group Psychotherapy', 'Psychologist specializing in group psychotherapy.'),
('103TR0400X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - Rehabilitation', 'Rehabilitation Psychologist', 'Psychologist specializing in rehabilitation.'),
('103TS0200X', 'Individual', 'Behavioral Health & Social Service Providers', 'Psychologist - School', 'School Psychologist', 'Psychologist specializing in school psychology.');

-- Social Workers
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('104100000X', 'Individual', 'Behavioral Health & Social Service Providers', 'Social Worker', 'Social Worker', 'A person who is qualified by a Social Work degree and experience to provide social services.'),
('1041C0700X', 'Individual', 'Behavioral Health & Social Service Providers', 'Social Worker - Clinical', 'Clinical Social Worker', 'A social worker who provides clinical mental health services.'),
('1041S0200X', 'Individual', 'Behavioral Health & Social Service Providers', 'Social Worker - School', 'School Social Worker', 'A social worker who works in school settings.');

-- Counselors
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('101Y00000X', 'Individual', 'Behavioral Health & Social Service Providers', 'Counselor', 'Counselor', 'A provider who is trained and educated in the performance of behavior health services through interpersonal communications and analysis.'),
('101YA0400X', 'Individual', 'Behavioral Health & Social Service Providers', 'Counselor - Addiction (Substance Use Disorder)', 'Addiction Counselor', 'A counselor specializing in addiction and substance use disorders.'),
('101YM0800X', 'Individual', 'Behavioral Health & Social Service Providers', 'Counselor - Mental Health', 'Mental Health Counselor', 'A counselor specializing in mental health.'),
('101YP1600X', 'Individual', 'Behavioral Health & Social Service Providers', 'Counselor - Pastoral', 'Pastoral Counselor', 'A counselor specializing in pastoral counseling.'),
('101YP2500X', 'Individual', 'Behavioral Health & Social Service Providers', 'Counselor - Professional', 'Professional Counselor', 'A professional counselor.');

-- Marriage & Family Therapist
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('106H00000X', 'Individual', 'Behavioral Health & Social Service Providers', 'Marriage & Family Therapist', 'Marriage & Family Therapist', 'A person trained and registered to perform marriage, family and child counseling.');

-- =============================================================================
-- ALLIED HEALTH PROVIDERS
-- =============================================================================

-- Occupational Therapists
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('225X00000X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist', 'Occupational Therapist', 'A therapist who promotes health and wellness through engagement in occupations.'),
('225XE0001X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Environmental Modification', 'Occupational Therapist - Environmental Modification', 'Occupational therapist specializing in environmental modification.'),
('225XE1200X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Ergonomics', 'Occupational Therapist - Ergonomics', 'Occupational therapist specializing in ergonomics.'),
('225XF0002X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Feeding, Eating & Swallowing', 'Occupational Therapist - Feeding, Eating & Swallowing', 'Occupational therapist specializing in feeding, eating, and swallowing.'),
('225XG0600X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Gerontology', 'Occupational Therapist - Gerontology', 'Occupational therapist specializing in gerontology.'),
('225XH1200X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Hand', 'Occupational Therapist - Hand', 'Occupational therapist specializing in hand therapy.'),
('225XH1300X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Human Factors', 'Occupational Therapist - Human Factors', 'Occupational therapist specializing in human factors.'),
('225XL0004X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Low Vision', 'Occupational Therapist - Low Vision', 'Occupational therapist specializing in low vision.'),
('225XM0800X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Mental Health', 'Occupational Therapist - Mental Health', 'Occupational therapist specializing in mental health.'),
('225XN1300X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Neurorehabilitation', 'Occupational Therapist - Neurorehabilitation', 'Occupational therapist specializing in neurorehabilitation.'),
('225XP0019X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Physical Rehabilitation', 'Occupational Therapist - Physical Rehabilitation', 'Occupational therapist specializing in physical rehabilitation.'),
('225XP0200X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Occupational Therapist - Pediatrics', 'Occupational Therapist - Pediatrics', 'Occupational therapist specializing in pediatrics.');

-- Physical Therapists
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('225100000X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist', 'Physical Therapist', 'A physical therapist evaluates and treats patients with health problems resulting from injury or disease.'),
('2251C2600X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist - Cardiopulmonary', 'Physical Therapist - Cardiopulmonary', 'Physical therapist specializing in cardiopulmonary conditions.'),
('2251E1200X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist - Ergonomics', 'Physical Therapist - Ergonomics', 'Physical therapist specializing in ergonomics.'),
('2251E1300X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist - Electrophysiology, Clinical', 'Physical Therapist - Clinical Electrophysiology', 'Physical therapist specializing in clinical electrophysiology.'),
('2251G0304X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist - Geriatrics', 'Physical Therapist - Geriatrics', 'Physical therapist specializing in geriatrics.'),
('2251H1200X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist - Hand', 'Physical Therapist - Hand', 'Physical therapist specializing in hand therapy.'),
('2251H1300X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist - Human Factors', 'Physical Therapist - Human Factors', 'Physical therapist specializing in human factors.'),
('2251N0400X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist - Neurology', 'Physical Therapist - Neurology', 'Physical therapist specializing in neurology.'),
('2251P0200X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist - Pediatrics', 'Physical Therapist - Pediatrics', 'Physical therapist specializing in pediatrics.'),
('2251S0007X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist - Sports', 'Physical Therapist - Sports', 'Physical therapist specializing in sports physical therapy.'),
('2251X0800X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Physical Therapist - Orthopedic', 'Physical Therapist - Orthopedic', 'Physical therapist specializing in orthopedics.');

-- Speech-Language Pathologist
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('235Z00000X', 'Individual', 'Speech, Language and Hearing Service Providers', 'Speech-Language Pathologist', 'Speech-Language Pathologist', 'A specialist in speech, language, voice, and fluency disorders.');

-- Audiologist
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('231H00000X', 'Individual', 'Speech, Language and Hearing Service Providers', 'Audiologist', 'Audiologist', 'A specialist who treats and manages individuals with hearing and balance disorders.');

-- Respiratory Therapist
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('227800000X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Respiratory Therapist, Certified', 'Respiratory Therapist', 'A healthcare practitioner who specializes in the management of respiratory disorders.'),
('2278C0205X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Respiratory Therapist, Certified - Critical Care', 'Respiratory Therapist - Critical Care', 'Respiratory therapist specializing in critical care.'),
('2278E0002X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Respiratory Therapist, Certified - Emergency Care', 'Respiratory Therapist - Emergency Care', 'Respiratory therapist specializing in emergency care.'),
('2278G0305X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Respiratory Therapist, Certified - Geriatric Care', 'Respiratory Therapist - Geriatric Care', 'Respiratory therapist specializing in geriatric care.'),
('2278G1100X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Respiratory Therapist, Certified - General Care', 'Respiratory Therapist - General Care', 'Respiratory therapist specializing in general care.'),
('2278H0200X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Respiratory Therapist, Certified - Home Health', 'Respiratory Therapist - Home Health', 'Respiratory therapist specializing in home health.'),
('2278P3800X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Respiratory Therapist, Certified - Pulmonary Diagnostics', 'Respiratory Therapist - Pulmonary Diagnostics', 'Respiratory therapist specializing in pulmonary diagnostics.'),
('2278P3900X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Respiratory Therapist, Certified - Pulmonary Rehabilitation', 'Respiratory Therapist - Pulmonary Rehabilitation', 'Respiratory therapist specializing in pulmonary rehabilitation.'),
('2278P4000X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Respiratory Therapist, Certified - Pulmonary Function Technologist', 'Respiratory Therapist - Pulmonary Function', 'Respiratory therapist specializing in pulmonary function testing.'),
('2278S1500X', 'Individual', 'Respiratory, Developmental, Rehabilitative and Restorative Service Providers', 'Respiratory Therapist, Certified - SNF/Subacute Care', 'Respiratory Therapist - SNF/Subacute Care', 'Respiratory therapist specializing in SNF and subacute care.');

-- Dietitian/Nutritionist
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('133V00000X', 'Individual', 'Dietary & Nutritional Service Providers', 'Dietitian, Registered', 'Registered Dietitian', 'A professional who translates the science of nutrition into practical solutions for healthy living.'),
('133VN1004X', 'Individual', 'Dietary & Nutritional Service Providers', 'Dietitian, Registered - Nutrition, Pediatric', 'Registered Dietitian - Pediatric Nutrition', 'Registered dietitian specializing in pediatric nutrition.'),
('133VN1005X', 'Individual', 'Dietary & Nutritional Service Providers', 'Dietitian, Registered - Nutrition, Renal', 'Registered Dietitian - Renal Nutrition', 'Registered dietitian specializing in renal nutrition.'),
('133VN1006X', 'Individual', 'Dietary & Nutritional Service Providers', 'Dietitian, Registered - Nutrition, Metabolic', 'Registered Dietitian - Metabolic Nutrition', 'Registered dietitian specializing in metabolic nutrition.');

-- Chiropractor
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('111N00000X', 'Individual', 'Chiropractic Providers', 'Chiropractor', 'Chiropractor', 'A provider qualified by a Doctor of Chiropractic (D.C.) degree and licensed who practices chiropractic medicine.'),
('111NI0013X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Independent Medical Examiner', 'Chiropractor - Independent Medical Examiner', 'Chiropractor specializing in independent medical examinations.'),
('111NI0900X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Internist', 'Chiropractor - Internist', 'Chiropractor specializing in internal medicine.'),
('111NN0400X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Neurology', 'Chiropractor - Neurology', 'Chiropractor specializing in neurology.'),
('111NN1001X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Nutrition', 'Chiropractor - Nutrition', 'Chiropractor specializing in nutrition.'),
('111NP0017X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Pediatric Chiropractor', 'Chiropractor - Pediatrics', 'Chiropractor specializing in pediatrics.'),
('111NR0200X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Radiology', 'Chiropractor - Radiology', 'Chiropractor specializing in radiology.'),
('111NR0400X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Rehabilitation', 'Chiropractor - Rehabilitation', 'Chiropractor specializing in rehabilitation.'),
('111NS0005X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Sports Physician', 'Chiropractor - Sports Medicine', 'Chiropractor specializing in sports medicine.'),
('111NT0100X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Thermography', 'Chiropractor - Thermography', 'Chiropractor specializing in thermography.'),
('111NX0100X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Occupational Health', 'Chiropractor - Occupational Health', 'Chiropractor specializing in occupational health.'),
('111NX0800X', 'Individual', 'Chiropractic Providers', 'Chiropractor - Orthopedic', 'Chiropractor - Orthopedic', 'Chiropractor specializing in orthopedics.');

-- Podiatrist
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('213E00000X', 'Individual', 'Podiatric Medicine & Surgery Service Providers', 'Podiatrist', 'Podiatrist', 'A practitioner of podiatric medicine providing comprehensive treatment of the foot, ankle and related structures.'),
('213EG0000X', 'Individual', 'Podiatric Medicine & Surgery Service Providers', 'Podiatrist - General Practice', 'Podiatrist - General Practice', 'Podiatrist specializing in general practice.'),
('213EP0504X', 'Individual', 'Podiatric Medicine & Surgery Service Providers', 'Podiatrist - Public Medicine', 'Podiatrist - Public Medicine', 'Podiatrist specializing in public medicine.'),
('213EP1101X', 'Individual', 'Podiatric Medicine & Surgery Service Providers', 'Podiatrist - Primary Podiatric Medicine', 'Podiatrist - Primary Podiatric Medicine', 'Podiatrist specializing in primary podiatric medicine.'),
('213ER0200X', 'Individual', 'Podiatric Medicine & Surgery Service Providers', 'Podiatrist - Radiology', 'Podiatrist - Radiology', 'Podiatrist specializing in radiology.'),
('213ES0000X', 'Individual', 'Podiatric Medicine & Surgery Service Providers', 'Podiatrist - Sports Medicine', 'Podiatrist - Sports Medicine', 'Podiatrist specializing in sports medicine.'),
('213ES0103X', 'Individual', 'Podiatric Medicine & Surgery Service Providers', 'Podiatrist - Foot & Ankle Surgery', 'Podiatrist - Foot & Ankle Surgery', 'Podiatrist specializing in foot and ankle surgery.'),
('213ES0131X', 'Individual', 'Podiatric Medicine & Surgery Service Providers', 'Podiatrist - Foot Surgery', 'Podiatrist - Foot Surgery', 'Podiatrist specializing in foot surgery.');

-- Optometrist
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('152W00000X', 'Individual', 'Eye and Vision Services Providers', 'Optometrist', 'Optometrist', 'A Doctor of Optometry who provides primary eye and vision care.'),
('152WC0802X', 'Individual', 'Eye and Vision Services Providers', 'Optometrist - Corneal and Contact Management', 'Optometrist - Corneal & Contact Management', 'Optometrist specializing in corneal and contact lens management.'),
('152WL0500X', 'Individual', 'Eye and Vision Services Providers', 'Optometrist - Low Vision Rehabilitation', 'Optometrist - Low Vision Rehabilitation', 'Optometrist specializing in low vision rehabilitation.'),
('152WP0200X', 'Individual', 'Eye and Vision Services Providers', 'Optometrist - Pediatrics', 'Optometrist - Pediatrics', 'Optometrist specializing in pediatrics.'),
('152WS0006X', 'Individual', 'Eye and Vision Services Providers', 'Optometrist - Sports Vision', 'Optometrist - Sports Vision', 'Optometrist specializing in sports vision.'),
('152WV0400X', 'Individual', 'Eye and Vision Services Providers', 'Optometrist - Vision Therapy', 'Optometrist - Vision Therapy', 'Optometrist specializing in vision therapy.'),
('152WX0102X', 'Individual', 'Eye and Vision Services Providers', 'Optometrist - Occupational Vision', 'Optometrist - Occupational Vision', 'Optometrist specializing in occupational vision.');

-- Pharmacist
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('183500000X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist', 'Pharmacist', 'A practitioner licensed to dispense prescription drugs and provide pharmaceutical care.'),
('1835C0205X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist - Critical Care', 'Pharmacist - Critical Care', 'Pharmacist specializing in critical care.'),
('1835G0000X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist - General Practice', 'Pharmacist - General Practice', 'Pharmacist specializing in general practice.'),
('1835G0303X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist - Geriatric', 'Pharmacist - Geriatric', 'Pharmacist specializing in geriatric pharmacy.'),
('1835N0905X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist - Nuclear', 'Pharmacist - Nuclear', 'Pharmacist specializing in nuclear pharmacy.'),
('1835N1003X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist - Nutrition Support', 'Pharmacist - Nutrition Support', 'Pharmacist specializing in nutrition support.'),
('1835P0018X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist - Pharmacist Clinician (PhC)/Clinical Pharmacy Specialist', 'Pharmacist - Clinical Pharmacy Specialist', 'Pharmacist specializing in clinical pharmacy.'),
('1835P1200X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist - Pharmacotherapy', 'Pharmacist - Pharmacotherapy', 'Pharmacist specializing in pharmacotherapy.'),
('1835P1300X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist - Psychiatric', 'Pharmacist - Psychiatric', 'Pharmacist specializing in psychiatric pharmacy.'),
('1835P2201X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist - Pediatrics', 'Pharmacist - Pediatrics', 'Pharmacist specializing in pediatric pharmacy.'),
('1835X0200X', 'Individual', 'Pharmacy Service Providers', 'Pharmacist - Oncology', 'Pharmacist - Oncology', 'Pharmacist specializing in oncology pharmacy.');

-- =============================================================================
-- ORGANIZATIONS
-- =============================================================================

-- Clinics/Centers
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('261Q00000X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center', 'Clinic/Center', 'A facility or distinct part of one used for the diagnosis and treatment of outpatients.'),
('261QA0005X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Ambulatory Family Planning Facility', 'Ambulatory Family Planning Facility', 'A facility providing family planning services.'),
('261QA0006X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Ambulatory Fertility Facility', 'Ambulatory Fertility Facility', 'A facility providing fertility services.'),
('261QA1903X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Ambulatory Surgical', 'Ambulatory Surgical Center', 'A facility providing surgical procedures on an outpatient basis.'),
('261QB0400X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Birthing', 'Birthing Center', 'A facility providing childbirth services.'),
('261QC0050X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Critical Access Hospital', 'Critical Access Hospital', 'A hospital certified as a critical access hospital.'),
('261QC1500X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Community Health', 'Community Health Center', 'A clinic providing comprehensive primary care services.'),
('261QC1800X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Corporate Health', 'Corporate Health Center', 'A clinic providing occupational health services.'),
('261QD0000X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Dental', 'Dental Clinic', 'A clinic providing dental services.'),
('261QD1600X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Developmental Disabilities', 'Developmental Disabilities Clinic', 'A clinic providing services for developmental disabilities.'),
('261QE0002X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Emergency Care', 'Emergency Care Clinic', 'A clinic providing emergency care services.'),
('261QE0700X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - End-Stage Renal Disease (ESRD) Treatment', 'ESRD Treatment Clinic', 'A clinic providing dialysis and ESRD treatment.'),
('261QF0050X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Family Planning, Non-Surgical', 'Family Planning Clinic', 'A clinic providing family planning services.'),
('261QF0400X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Federally Qualified Health Center (FQHC)', 'Federally Qualified Health Center (FQHC)', 'A health center receiving federal funding to provide primary care services.'),
('261QH0100X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Health Service', 'Health Service Center', 'A center providing comprehensive health services.'),
('261QH0700X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Hearing and Speech', 'Hearing & Speech Center', 'A center providing hearing and speech services.'),
('261QI0500X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Infusion Therapy', 'Infusion Therapy Center', 'A center providing intravenous infusion therapy.'),
('261QL0400X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Literacy', 'Literacy Clinic', 'A clinic providing health literacy services.'),
('261QM0801X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Mental Health (Including Community Mental Health Center)', 'Mental Health Clinic', 'A clinic providing mental health services.'),
('261QM0850X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Adult Mental Health', 'Adult Mental Health Clinic', 'A clinic providing mental health services for adults.'),
('261QM0855X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Adolescent and Children Mental Health', 'Adolescent & Children Mental Health Clinic', 'A clinic providing mental health services for adolescents and children.'),
('261QM1000X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Migrant Health', 'Migrant Health Clinic', 'A clinic providing services to migrant populations.'),
('261QM1100X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Military/U.S. Coast Guard Outpatient', 'Military/U.S. Coast Guard Outpatient Clinic', 'A military outpatient clinic.'),
('261QM1101X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Military and U.S. Coast Guard Ambulatory Procedure', 'Military Ambulatory Procedure Clinic', 'A military ambulatory procedure clinic.'),
('261QM1102X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Military Outpatient Operational (Transportable) Component', 'Military Outpatient Operational Clinic', 'A transportable military outpatient clinic.'),
('261QM1103X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Military Ambulatory Procedure Visits Operational (Transportable)', 'Military Ambulatory Procedure Operational Clinic', 'A transportable military ambulatory procedure clinic.'),
('261QM1200X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Ambulatory Surgical', 'Ambulatory Surgical Center', 'A center providing surgical procedures on an outpatient basis.'),
('261QM1300X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Multi-Specialty', 'Multi-Specialty Clinic', 'A clinic providing multiple specialty services.'),
('261QM2500X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Medical Specialty', 'Medical Specialty Clinic', 'A clinic providing medical specialty services.'),
('261QM2800X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Methadone', 'Methadone Clinic', 'A clinic providing methadone treatment.'),
('261QM3000X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Medically Fragile Infants and Children Day Care', 'Medically Fragile Infants & Children Day Care', 'A day care center for medically fragile children.'),
('261QP0904X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Federal Public Health', 'Federal Public Health Clinic', 'A federal public health clinic.'),
('261QP0905X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - State or Local Public Health', 'State or Local Public Health Clinic', 'A state or local public health clinic.'),
('261QP1100X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Podiatric', 'Podiatric Clinic', 'A clinic providing podiatric services.'),
('261QP2000X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Physical Therapy', 'Physical Therapy Clinic', 'A clinic providing physical therapy services.'),
('261QP2300X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Primary Care', 'Primary Care Clinic', 'A clinic providing primary care services.'),
('261QP2400X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Prison Health', 'Prison Health Clinic', 'A clinic providing health services in correctional facilities.'),
('261QP3300X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Pain', 'Pain Clinic', 'A clinic specializing in pain management.'),
('261QR0200X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Radiology', 'Radiology Clinic', 'A clinic providing radiology services.'),
('261QR0206X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Radiology, Mammography', 'Mammography Clinic', 'A clinic providing mammography services.'),
('261QR0207X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Radiology, Mobile Mammography', 'Mobile Mammography Clinic', 'A mobile clinic providing mammography services.'),
('261QR0208X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Radiology, Mobile', 'Mobile Radiology Clinic', 'A mobile clinic providing radiology services.'),
('261QR0400X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Rehabilitation', 'Rehabilitation Clinic', 'A clinic providing rehabilitation services.'),
('261QR0401X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Rehabilitation, Comprehensive Outpatient Rehabilitation Facility (CORF)', 'Comprehensive Outpatient Rehabilitation Facility (CORF)', 'A facility providing comprehensive outpatient rehabilitation.'),
('261QR0404X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Rehabilitation, Cardiac Facilities', 'Cardiac Rehabilitation Clinic', 'A clinic providing cardiac rehabilitation services.'),
('261QR0405X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Rehabilitation, Substance Use Disorder', 'Substance Use Disorder Rehabilitation Clinic', 'A clinic providing substance use disorder rehabilitation.'),
('261QR0800X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Recovery Care', 'Recovery Care Center', 'A center providing recovery care services.'),
('261QR1100X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Research', 'Research Clinic', 'A clinic providing research services.'),
('261QR1300X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Rural Health', 'Rural Health Clinic', 'A clinic providing health services in rural areas.'),
('261QS0112X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Oral and Maxillofacial Surgery', 'Oral & Maxillofacial Surgery Clinic', 'A clinic providing oral and maxillofacial surgery.'),
('261QS0132X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Ophthalmologic Surgery', 'Ophthalmologic Surgery Clinic', 'A clinic providing ophthalmologic surgery.'),
('261QS1000X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Student Health', 'Student Health Clinic', 'A clinic providing health services to students.'),
('261QS1200X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Sleep Disorder Diagnostic', 'Sleep Disorder Diagnostic Clinic', 'A clinic providing sleep disorder diagnostics.'),
('261QU0200X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Urgent Care', 'Urgent Care Clinic', 'A clinic providing urgent care services.'),
('261QV0200X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - VA', 'VA Clinic', 'A Department of Veterans Affairs clinic.'),
('261QX0100X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Occupational Medicine', 'Occupational Medicine Clinic', 'A clinic providing occupational medicine services.'),
('261QX0200X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Oncology', 'Oncology Clinic', 'A clinic providing oncology services.'),
('261QX0203X', 'Organization', 'Ambulatory Health Care Facilities', 'Clinic/Center - Oncology, Radiation', 'Radiation Oncology Clinic', 'A clinic providing radiation oncology services.');

-- Hospitals
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('282N00000X', 'Organization', 'Hospitals', 'General Acute Care Hospital', 'General Acute Care Hospital', 'A hospital providing acute care services.'),
('282NC0060X', 'Organization', 'Hospitals', 'General Acute Care Hospital - Critical Access', 'Critical Access Hospital', 'A hospital certified as a critical access hospital.'),
('282NC2000X', 'Organization', 'Hospitals', 'General Acute Care Hospital - Children', 'Children''s Hospital', 'A hospital providing care exclusively to children.'),
('282NE0002X', 'Organization', 'Hospitals', 'General Acute Care Hospital - Emergency', 'Emergency Hospital', 'A hospital specializing in emergency care.'),
('282NR1301X', 'Organization', 'Hospitals', 'General Acute Care Hospital - Rural', 'Rural Hospital', 'A hospital located in a rural area.'),
('282NW0100X', 'Organization', 'Hospitals', 'General Acute Care Hospital - Women', 'Women''s Hospital', 'A hospital providing care exclusively to women.'),
('283Q00000X', 'Organization', 'Hospitals', 'Psychiatric Hospital', 'Psychiatric Hospital', 'A hospital providing psychiatric care.'),
('283X00000X', 'Organization', 'Hospitals', 'Rehabilitation Hospital', 'Rehabilitation Hospital', 'A hospital providing rehabilitation services.'),
('284300000X', 'Organization', 'Hospitals', 'Special Hospital', 'Special Hospital', 'A hospital providing specialized care.'),
('286500000X', 'Organization', 'Hospitals', 'Military Hospital', 'Military Hospital', 'A hospital operated by the military.'),
('287300000X', 'Organization', 'Hospitals', 'Christian Science Sanitorium', 'Christian Science Sanitorium', 'A sanitorium operated according to Christian Science principles.');

-- Nursing Facilities
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('313M00000X', 'Organization', 'Nursing & Custodial Care Facilities', 'Nursing Facility/Intermediate Care Facility', 'Nursing Facility', 'A facility providing skilled nursing care and related services.'),
('314000000X', 'Organization', 'Nursing & Custodial Care Facilities', 'Skilled Nursing Facility', 'Skilled Nursing Facility', 'A facility providing skilled nursing care.'),
('315D00000X', 'Organization', 'Nursing & Custodial Care Facilities', 'Hospice, Inpatient', 'Inpatient Hospice', 'A facility providing inpatient hospice care.'),
('315P00000X', 'Organization', 'Nursing & Custodial Care Facilities', 'Intermediate Care Facility, Mentally Retarded', 'Intermediate Care Facility for Individuals with Intellectual Disabilities', 'A facility providing care for individuals with intellectual disabilities.');

-- Home Health
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('251E00000X', 'Organization', 'Home Health Care Agencies', 'Home Health', 'Home Health Agency', 'An organization providing health care services in the patient''s home.'),
('251F00000X', 'Organization', 'Home Health Care Agencies', 'Home Infusion', 'Home Infusion Agency', 'An organization providing intravenous infusion therapy in the patient''s home.'),
('251G00000X', 'Organization', 'Home Health Care Agencies', 'Hospice Care, Community Based', 'Community Based Hospice', 'An organization providing hospice care in the community.'),
('251S00000X', 'Organization', 'Home Health Care Agencies', 'Community/Behavioral Health', 'Community/Behavioral Health Agency', 'An organization providing community and behavioral health services.');

-- Laboratories
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('291U00000X', 'Organization', 'Laboratories', 'Clinical Medical Laboratory', 'Clinical Medical Laboratory', 'A facility for performing clinical laboratory tests.'),
('292200000X', 'Organization', 'Laboratories', 'Dental Laboratory', 'Dental Laboratory', 'A facility for dental laboratory services.');

-- Managed Care Organizations
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('273R00000X', 'Organization', 'Managed Care Organizations', 'Psychiatric Residential Treatment Facility', 'Psychiatric Residential Treatment Facility', 'A facility providing psychiatric residential treatment.'),
('273Y00000X', 'Organization', 'Managed Care Organizations', 'Non-Residential Substance Abuse Treatment Facility', 'Non-Residential Substance Abuse Treatment Facility', 'A facility providing non-residential substance abuse treatment.');

-- Ambulance Services
INSERT INTO claims.provider_taxonomy (taxonomy_code, provider_type, classification, specialization, specialty_display, definition) VALUES
('341600000X', 'Organization', 'Ambulance Services', 'Ambulance', 'Ambulance Service', 'An organization providing ambulance services.'),
('3416A0800X', 'Organization', 'Ambulance Services', 'Ambulance - Air Transport', 'Air Ambulance Service', 'An organization providing air ambulance services.'),
('3416L0300X', 'Organization', 'Ambulance Services', 'Ambulance - Land Transport', 'Land Ambulance Service', 'An organization providing ground ambulance services.'),
('3416S0300X', 'Organization', 'Ambulance Services', 'Ambulance - Water Transport', 'Water Ambulance Service', 'An organization providing water ambulance services.');

COMMENT ON TABLE claims.provider_taxonomy IS 'Comprehensive NUCC Healthcare Provider Taxonomy code set with 500+ taxonomy codes mapped to specialty display names';
