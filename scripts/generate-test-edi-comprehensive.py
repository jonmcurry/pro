#!/usr/bin/env python3
"""
Generate comprehensive test EDI 837p files with Phases 4, 5, and 6 data elements.

Includes:
- Phase 4: Coordination of Benefits (Loop 2320 - SBR, CAS, OI, MOA)
- Phase 5: Specialized claim types (CR1, CR2, CR3, CR5, CR7, HSD, PWK)
- Phase 6: Additional loops (2010BC patient, 2420B purchased service, MEA test results, HCP repricing)
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# Read master data
def load_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

# Sample diagnoses (ICD-10) with medical necessity for various claim types
DIAGNOSES = {
    'general': ['E78.5', 'I10', 'E11.9', 'J06.9', 'M79.3', 'R10.9', 'F41.9', 'M54.5', 'K21.9', 'Z23'],
    'ambulance': ['I21.9', 'S06.0X0A', 'R07.9', 'I63.9'],  # AMI, head injury, chest pain, stroke
    'chiropractic': ['M54.5', 'M54.2', 'M99.03', 'M53.2X7A'],  # Low back pain, cervical pain, subluxation
    'dme': ['E11.65', 'J44.1', 'Z99.81', 'I50.9'],  # Diabetes with hyperglycemia, COPD, oxygen dependence
    'home_health': ['I69.354', 'M62.81', 'Z96.1', 'I48.91'],  # Stroke with hemiplegia, muscle weakness
    'pregnancy': ['O09.90', 'Z34.90', 'O80'],  # Pregnancy supervision
}

# Sample procedures (CPT codes) by claim type
PROCEDURES = {
    'general': [
        ('99213', 'Office visit established', 85.00),
        ('99214', 'Office visit detailed', 125.00),
        ('99215', 'Office visit comprehensive', 170.00),
        ('36415', 'Venipuncture', 22.00),
        ('85025', 'Complete blood count', 45.00),
        ('80053', 'Comprehensive metabolic panel', 55.00),
        ('93000', 'ECG', 65.00),
        ('71046', 'Chest X-ray', 95.00),
    ],
    'ambulance': [
        ('A0429', 'Ambulance BLS emergency', 450.00),
        ('A0427', 'Ambulance ALS emergency', 650.00),
    ],
    'chiropractic': [
        ('98940', 'Chiropractic manipulation 1-2 regions', 55.00),
        ('98941', 'Chiropractic manipulation 3-4 regions', 75.00),
    ],
    'dme': [
        ('E0601', 'CPAP device', 850.00),
        ('E0424', 'Stationary oxygen system rental', 125.00),
        ('E1390', 'Oxygen concentrator', 275.00),
    ],
    'home_health': [
        ('G0151', 'Home health PT services', 150.00),
        ('G0152', 'Home health OT services', 150.00),
        ('G0154', 'Home health skilled nursing', 180.00),
    ],
}

# Payers (primary and secondary)
PAYERS = [
    ('87726', 'Medicare'),
    ('HUMN001', 'Humana'),
    ('UHC0001', 'UnitedHealthcare'),
    ('AETNA01', 'Aetna'),
    ('CIGNA01', 'Cigna'),
    ('BCBS001', 'Blue Cross Blue Shield'),
]

# Patient names with demographics
PATIENTS = [
    ('Smith', 'John', 'M', '19750315', 185),  # Weight in lbs
    ('Johnson', 'Mary', 'F', '19820622', 142),
    ('Williams', 'Robert', 'M', '19691204', 195),
    ('Brown', 'Patricia', 'F', '19580918', 155),
    ('Jones', 'Michael', 'M', '19920503', 178),
    ('Garcia', 'Jennifer', 'F', '19880827', 138),
    ('Miller', 'David', 'M', '19630711', 210),
    ('Davis', 'Linda', 'F', '19750129', 148),
    ('Rodriguez', 'James', 'M', '19911015', 172),
    ('Martinez', 'Barbara', 'F', '19670408', 162),
]

# Dependent children (for patient table - Loop 2010BC)
DEPENDENTS = [
    ('Smith', 'Emma', 'F', '20150410', 45),
    ('Johnson', 'Liam', 'M', '20180715', 38),
    ('Williams', 'Olivia', 'F', '20120228', 72),
    ('Brown', 'Noah', 'M', '20160903', 52),
    ('Jones', 'Ava', 'F', '20190521', 35),
]

# Reference labs (for Loop 2420B)
REFERENCE_LABS = [
    ('1234567890', 'Quest Diagnostics', 35.00),
    ('0987654321', 'LabCorp', 38.00),
    ('1122334455', 'BioReference Laboratories', 42.00),
]

# Subluxation levels for chiropractic
SUBLUXATION_LEVELS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'T1', 'T2', 'T5', 'T10', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1']

def generate_isa_segment(sender_id, receiver_id, control_num, date_str, time_str):
    """Generate ISA (Interchange Control Header) segment"""
    return f"ISA*00*          *00*          *ZZ*{sender_id:<15}*ZZ*{receiver_id:<15}*{date_str}*{time_str}*^*00501*{control_num:>09}*0*P*:~"

def generate_gs_segment(sender_id, receiver_id, date_str, time_str, group_num):
    """Generate GS (Functional Group Header) segment"""
    return f"GS*HC*{sender_id}*{receiver_id}*{date_str}*{time_str}*{group_num}*X*005010X222A1~"

def generate_cob_loop(payer_sequence, primary_paid_amount):
    """Generate Loop 2320 - Coordination of Benefits (Phase 4)"""
    lines = []

    # SBR segment - Other Subscriber Information
    # P=Primary (for secondary claims), S=Secondary (for tertiary claims)
    payer_code, payer_name = random.choice(PAYERS[:3])  # Use Medicare, Humana, UHC as other payers

    lines.append(f"SBR*{payer_sequence}*18*GROUP123*ACME INC*12***CI~")

    # CAS segment - Claim Adjustment (Phase 4)
    # CO=Contractual Obligation, PR=Patient Responsibility, etc.
    adjustment_groups = [
        ('CO', '45', 25.50),  # Contractual adjustment - Charge exceeds fee schedule
        ('PR', '1', 15.00),   # Patient Responsibility - Deductible
        ('CO', '97', 10.00),  # Contractual adjustment - Bundled service
    ]

    for group_code, reason_code, amount in random.sample(adjustment_groups, random.randint(1, 2)):
        lines.append(f"CAS*{group_code}*{reason_code}*{amount:.2f}~")

    # OI segment - Other Insurance Coverage Information (Phase 4)
    lines.append("OI***Y*P**Y~")  # Benefits assignment cert=Y, Patient signature source=P, Release of info=Y

    # MOA segment - Medicare Outpatient Adjudication (Phase 4) - only for Medicare
    if payer_name == 'Medicare':
        lines.append(f"MOA*{primary_paid_amount:.2f}*45.00*MA01~")  # Reimbursement rate, HCPCS payable, remark code

    # AMT segment - Other payer paid amount
    lines.append(f"AMT*D*{primary_paid_amount:.2f}~")

    # Payer name (NM1*PR)
    lines.append(f"NM1*PR*2*{payer_name}****PI*{payer_code}~")

    return lines

def generate_ambulance_segments(claim_id, dos_date):
    """Generate ambulance-specific segments (Phase 5.1 - CR1, Loops 2310E/F)"""
    lines = []

    # CR1 segment - Ambulance Transport Information
    ambulance_reasons = ['A', 'B', 'C', 'D', 'E']  # A=Patient transported, B=Patient condition
    distance = random.uniform(5.0, 45.0)
    patient_weight = random.randint(120, 250)

    lines.append(f"CR1*LB*{random.uniform(100, 300):.2f}*{random.choice(ambulance_reasons)}**{distance:.1f}*{patient_weight}*1~")

    # Loop 2310E - Ambulance Pick-up Location
    lines.append("NM1*PW*2*ST MARY HOSPITAL****XX*1234567890~")
    lines.append("N3*1000 Hospital Drive~")
    lines.append("N4*Chicago*IL*60610~")

    # Loop 2310F - Ambulance Drop-off Location
    lines.append("NM1*45*2*RUSH MEDICAL CENTER****XX*0987654321~")
    lines.append("N3*1653 W Congress Pkwy~")
    lines.append("N4*Chicago*IL*60612~")

    return lines

def generate_chiropractic_segments():
    """Generate chiropractic-specific segments (Phase 5.4 - CR2)"""
    lines = []

    # CR2 segment - Spinal Manipulation Service Information
    manipulation_count = random.randint(1, 4)
    subluxation_1 = random.choice(SUBLUXATION_LEVELS)
    subluxation_2 = random.choice(SUBLUXATION_LEVELS) if random.random() > 0.5 else ''
    condition_codes = ['A', 'C', 'M', 'S']  # A=Acute, C=Chronic, M=Acute manifestation, S=Subsequent

    lines.append(f"CR2*{manipulation_count}*{random.choice(condition_codes)}**{subluxation_1}*{subluxation_2 if subluxation_2 else ''}~")

    return lines

def generate_dme_segments(procedure_code):
    """Generate DME-specific segments (Phase 5.2 - CR3)"""
    lines = []

    # CR3 segment - Durable Medical Equipment Certification
    certification_type = random.choice(['I', 'R', 'S'])  # I=Initial, R=Revised, S=Recertification
    duration_months = random.choice([1, 3, 6, 12])

    cert_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')

    lines.append(f"CR3*{certification_type}*MO*{duration_months}~")
    lines.append(f"DTP*471*D8*{cert_date}~")  # Certification revision date

    return lines

def generate_oxygen_therapy_segments():
    """Generate oxygen therapy segments (Phase 5.5 - CR5)"""
    lines = []

    # CR5 segment - Home Oxygen Therapy Information
    equipment_type = random.choice(['OXG', 'OXY', 'OXC'])  # OXG=Oxygen gas, OXY=Oxygen, OXC=Concentrator
    flow_rate = random.uniform(1.0, 5.0)
    daily_usage = random.uniform(12.0, 24.0)
    abg_value = random.uniform(55.0, 75.0)  # Arterial blood gas (PO2)
    o2_sat = random.randint(85, 92)  # Oxygen saturation %

    lines.append(f"CR5*{equipment_type}*{flow_rate:.1f}*{daily_usage:.1f}*{int(daily_usage)}~")
    lines.append(f"CR5*{equipment_type}**{abg_value:.1f}*{o2_sat}*R~")  # R=Rest condition

    test_date = (datetime.now() - timedelta(days=15)).strftime('%Y%m%d')
    lines.append(f"DTP*484*D8*{test_date}~")  # Oxygen test date

    return lines

def generate_home_health_segments():
    """Generate home health segments (Phase 5.3 - CR7, HSD)"""
    lines = []

    # CR7 segment - Home Health Care Plan Information
    disciplines = ['PT', 'OT', 'ST', 'SN']  # Physical Therapy, Occupational, Speech, Skilled Nursing
    discipline = random.choice(disciplines)
    total_visits = random.randint(10, 30)
    period_days = random.choice([30, 60, 90])
    prognosis = random.choice(['1', '2', '3', '4'])  # 1=Excellent, 2=Good, 3=Fair, 4=Poor

    lines.append(f"CR7*{discipline}*{total_visits}*{period_days}*DA*{prognosis}~")

    # HSD segment - Health Care Services Delivery
    frequency = random.randint(2, 5)  # Times per week
    lines.append(f"HSD*VS*{frequency}*WK*1~")  # 1=First week

    return lines

def generate_attachment_segment():
    """Generate PWK segment - Paperwork/Attachment (Phase 5.6)"""
    lines = []

    # PWK segment - Attachment Information
    report_types = ['03', 'OB', 'PY', 'RR', 'DA']  # 03=Justifying treatment, OB=Operative note, etc.
    transmission_codes = ['AA', 'AB', 'BM', 'EL']  # AA=Available on request, BM=By mail, EL=Electronic

    report_type = random.choice(report_types)
    transmission = random.choice(transmission_codes)
    control_num = f"ATT{random.randint(100000, 999999)}"

    lines.append(f"PWK*{report_type}*{transmission}***AC*{control_num}~")

    return lines

def generate_patient_loop(dependent_idx):
    """Generate Loop 2010BC - Patient Information (Phase 6.1) when patient != subscriber"""
    lines = []

    last_name, first_name, gender, dob, weight = DEPENDENTS[dependent_idx % len(DEPENDENTS)]

    # NM1*QC - Patient Name
    lines.append(f"NM1*QC*1*{last_name}*{first_name}~")

    # N3/N4 - Patient Address
    lines.append("N3*123 Dependent Lane~")
    lines.append(f"N4*Chicago*IL*{random.randint(60000, 60999)}~")

    # DMG - Patient Demographics
    pregnancy_ind = 'Y' if gender == 'F' and random.random() > 0.8 else 'N'
    lines.append(f"DMG*D8*{dob}*{gender}***{weight}***{pregnancy_ind}~")

    return lines

def generate_purchased_service_loop():
    """Generate Loop 2420B - Purchased Service Provider (Phase 6.2)"""
    lines = []

    lab_npi, lab_name, lab_charge = random.choice(REFERENCE_LABS)

    # NM1*QB - Purchased Service Provider
    lines.append(f"NM1*QB*2*{lab_name}****XX*{lab_npi}~")

    # AMT*KH - Purchased Service Charge
    lines.append(f"AMT*KH*{lab_charge:.2f}~")

    return lines

def generate_test_result_segments(procedure_code):
    """Generate MEA segments - Test Results (Phase 6.3)"""
    lines = []

    # Lab test results based on procedure
    if '85025' in procedure_code:  # CBC
        # Hemoglobin
        hgb_value = random.uniform(12.0, 16.0)
        lines.append(f"MEA*TR*HGB*{hgb_value:.1f}*g/dL*12.0*16.0*N~")

        # WBC
        wbc_value = random.uniform(4.5, 11.0)
        significance = 'N' if 4.5 <= wbc_value <= 11.0 else 'HI' if wbc_value > 11.0 else 'LO'
        lines.append(f"MEA*TR*WBC*{wbc_value:.1f}*K/uL*4.5*11.0*{significance}~")

    elif '80053' in procedure_code:  # Metabolic panel
        # Glucose
        glucose = random.uniform(70, 140)
        significance = 'N' if glucose <= 100 else 'HI'
        lines.append(f"MEA*TR*GLU*{glucose:.0f}*mg/dL*70*100*{significance}~")

        # Creatinine
        creat = random.uniform(0.7, 1.3)
        lines.append(f"MEA*TR*CREAT*{creat:.2f}*mg/dL*0.7*1.3*N~")

    return lines

def generate_repricing_segments(charge_amount):
    """Generate HCP segment - Repricing Information (Phase 6.4)"""
    lines = []

    # HCP segment - Claim/Line Pricing
    pricing_methodologies = ['01', '02', '03', '04']  # 01=As billed, 02=Fee schedule, 03=Contractual %
    pricing_method = random.choice(pricing_methodologies)

    repriced_amount = charge_amount * random.uniform(0.60, 0.85)  # 60-85% of billed
    savings = charge_amount - repriced_amount
    unit_price = repriced_amount  # For single unit

    lines.append(f"HCP*{pricing_method}*{repriced_amount:.2f}*{savings:.2f}***{unit_price:.2f}~")

    return lines

def generate_claim(claim_num, facility, provider, patient_idx, dos_date, claim_type='general', include_cob=False, include_dependent=False):
    """Generate a single claim with comprehensive data elements"""
    last_name, first_name, gender, dob, weight = PATIENTS[patient_idx % len(PATIENTS)]
    member_id = f"MEM{random.randint(100000000, 999999999)}"
    payer_code, payer_name = random.choice(PAYERS)

    # Select diagnosis and procedure based on claim type
    diagnosis = random.choice(DIAGNOSES.get(claim_type, DIAGNOSES['general']))
    procedures = PROCEDURES.get(claim_type, PROCEDURES['general'])
    procedure_code, procedure_desc, charge = random.choice(procedures)

    claim_lines = []

    # Subscriber level
    claim_lines.append(f"HL*{claim_num}*1*22*{'1' if include_dependent else '0'}~")

    # SBR segment - Subscriber relationship
    # For COB: S=Secondary, for regular: P=Primary
    sbr_code = 'S' if include_cob else 'P'
    claim_lines.append(f"SBR*{sbr_code}*18******CI~")

    # Subscriber name (NM1*IL)
    claim_lines.append(f"NM1*IL*1*{last_name}*{first_name}****MI*{member_id}~")
    claim_lines.append(f"N3*123 Patient St~")
    claim_lines.append(f"N4*Chicago*IL*{random.randint(60000, 60999)}~")
    claim_lines.append(f"DMG*D8*{dob}*{gender}~")

    # Payer name (NM1*PR)
    claim_lines.append(f"NM1*PR*2*{payer_name}****PI*{payer_code}~")

    # Loop 2320 - Other Subscriber (COB) if secondary claim (Phase 4)
    if include_cob:
        primary_paid = charge * random.uniform(0.50, 0.70)
        claim_lines.extend(generate_cob_loop('P', primary_paid))

    # Loop 2010BC - Patient Information (Phase 6.1) if dependent
    if include_dependent:
        claim_lines.append(f"HL*{claim_num + 1000}*{claim_num}*23*0~")
        claim_lines.append("PAT*19~")  # 19=Child
        claim_lines.extend(generate_patient_loop(patient_idx))

    # Claim information
    claim_id = f"{facility['facility_code']}-{claim_num:06d}"
    claim_lines.append(f"CLM*{claim_id}*{charge:.2f}***11:B:1*Y*A*Y*Y~")
    claim_lines.append(f"DTP*472*D8*{dos_date}~")
    claim_lines.append(f"REF*D9*{claim_id}~")

    # Additional REF segments (Phase 3)
    claim_lines.append(f"REF*9F*{claim_id}-AUTH~")  # Authorization number
    claim_lines.append(f"REF*F8*{payer_code}-ORIG~")  # Payer claim control number

    # HI segment - Diagnosis
    claim_lines.append(f"HI*ABK:{diagnosis}~")

    # NTE segment - Claim Note (Phase 3)
    if random.random() > 0.7:
        claim_lines.append("NTE*ADD*PATIENT REQUIRES SPECIAL HANDLING~")

    # CRC segment - Condition Codes (Phase 3)
    if random.random() > 0.8:
        claim_lines.append("CRC*07*Y*38~")  # Condition 07=Condition related to employment

    # Ambulance-specific segments (Phase 5.1)
    if claim_type == 'ambulance':
        claim_lines.extend(generate_ambulance_segments(claim_id, dos_date))

    # Chiropractic-specific segments (Phase 5.4)
    if claim_type == 'chiropractic':
        claim_lines.extend(generate_chiropractic_segments())

    # Home health-specific segments (Phase 5.3)
    if claim_type == 'home_health':
        claim_lines.extend(generate_home_health_segments())

    # PWK segment - Attachments (Phase 5.6)
    if random.random() > 0.6:
        claim_lines.extend(generate_attachment_segment())

    # Rendering provider (NM1*82)
    claim_lines.append(f"NM1*82*1*{provider['last_name']}*{provider['first_name']}****XX*{provider['provider_npi']}~")

    # PRV segment - Provider Specialty (Phase 3)
    claim_lines.append(f"PRV*PE*PXC*207Q00000X~")  # Family Medicine taxonomy

    # Service Facility (Loop 2310C)
    claim_lines.append(f"NM1*77*2*{facility['facility_name']}****XX*{facility['npi']}~")
    claim_lines.append(f"N3*{facility.get('address_line1', 'Medical Center Address')}~")
    claim_lines.append(f"N4*{facility.get('city', 'Chicago')}*{facility.get('state_code', 'IL')}*{facility.get('postal_code', '60601')}~")

    # HCP segment - Claim-level repricing (Phase 6.4)
    if random.random() > 0.5:
        claim_lines.extend(generate_repricing_segments(charge))

    # Service line
    claim_lines.append(f"LX*1~")
    claim_lines.append(f"SV1*HC:{procedure_code}*{charge:.2f}*UN*1***1~")
    claim_lines.append(f"DTP*472*D8*{dos_date}~")

    # DME-specific service line segments (Phase 5.2)
    if claim_type == 'dme':
        claim_lines.extend(generate_dme_segments(procedure_code))

    # Oxygen therapy service line segments (Phase 5.5)
    if 'E0424' in procedure_code or 'E1390' in procedure_code:
        claim_lines.extend(generate_oxygen_therapy_segments())

    # Loop 2420B - Purchased Service Provider (Phase 6.2) - for labs
    if '85025' in procedure_code or '80053' in procedure_code:
        claim_lines.extend(generate_purchased_service_loop())

    # MEA segments - Test Results (Phase 6.3)
    if '85025' in procedure_code or '80053' in procedure_code:
        claim_lines.extend(generate_test_result_segments(procedure_code))

    # AMT segments - Supplemental amounts (Phase 3)
    if random.random() > 0.7:
        claim_lines.append(f"AMT*T*{charge * 0.90:.2f}~")  # Approved amount
        claim_lines.append(f"AMT*A8*{charge * 0.10:.2f}~")  # Non-covered charges

    # HCP segment - Line-level repricing (Phase 6.4)
    if random.random() > 0.4:
        claim_lines.extend(generate_repricing_segments(charge))

    return claim_lines

def generate_edi_file(facility, providers, num_claims, output_path):
    """Generate comprehensive EDI 837p file"""
    date_str = datetime.now().strftime('%y%m%d')
    time_str = datetime.now().strftime('%H%M')

    sender_id = facility['facility_code']
    receiver_id = 'PAYER001'
    control_num = random.randint(1, 999999999)

    lines = []

    # ISA segment
    lines.append(generate_isa_segment(sender_id, receiver_id, control_num, date_str, time_str))

    # GS segment
    lines.append(generate_gs_segment(sender_id, receiver_id, '20' + date_str, time_str, 1))

    # ST segment
    lines.append("ST*837*0001*005010X222A1~")

    # BHT segment
    lines.append(f"BHT*0019*00*1*20{date_str}*{time_str}*CH~")

    # Submitter (Loop 1000A)
    lines.append(f"NM1*41*2*{facility['facility_name']}****46*{facility['facility_code']}~")
    lines.append("PER*IC*Contact Name*TE*5555551234~")

    # Receiver (Loop 1000B)
    lines.append(f"NM1*40*2*PAYER NAME****46*{receiver_id}~")

    # Billing provider (Loop 2000A)
    lines.append("HL*1**20*1~")
    lines.append(f"NM1*85*2*{facility['facility_name']}****XX*{facility['facility_code']}~")
    lines.append(f"N3*{facility.get('address_line1', 'Medical Center Address')}~")
    lines.append(f"N4*{facility.get('city', 'Chicago')}*{facility.get('state_code', 'IL')}*{facility.get('postal_code', '60601')}~")
    lines.append(f"REF*EI*{facility.get('tax_id', '123456789')}~")

    # Generate claims with variety of types
    claim_types_distribution = {
        'general': 0.70,      # 70% general claims
        'ambulance': 0.05,    # 5% ambulance
        'chiropractic': 0.05, # 5% chiropractic
        'dme': 0.08,          # 8% DME
        'home_health': 0.07,  # 7% home health
        'pregnancy': 0.05,    # 5% pregnancy (for dependent example)
    }

    for claim_idx in range(num_claims):
        provider = providers[claim_idx % len(providers)]

        # Random date of service in last 6 months
        days_ago = random.randint(1, 180)
        dos_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y%m%d')

        # Determine claim type
        rand_val = random.random()
        cumulative = 0
        claim_type = 'general'
        for ctype, prob in claim_types_distribution.items():
            cumulative += prob
            if rand_val <= cumulative:
                claim_type = ctype
                break

        # 20% chance of COB (secondary claims)
        include_cob = random.random() > 0.80

        # 15% chance of dependent (patient != subscriber)
        include_dependent = random.random() > 0.85

        claim_lines = generate_claim(
            claim_idx + 2,
            facility,
            provider,
            claim_idx,
            dos_date,
            claim_type=claim_type,
            include_cob=include_cob,
            include_dependent=include_dependent
        )
        lines.extend(claim_lines)

    # SE segment
    lines.append(f"SE*{len(lines) + 1}*0001~")

    # GE segment
    lines.append("GE*1*1~")

    # IEA segment
    lines.append(f"IEA*1*{control_num:>09}~")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Generated: {output_path}")
    print(f"  Facility: {facility['facility_code']} ({facility['facility_name']})")
    print(f"  NPI: {facility['npi']}")
    print(f"  Claims: {num_claims}")
    print(f"  Comprehensive data: Phases 4-6 elements included")

def main():
    # Paths
    setup_dir = Path('../test_data/setup')
    output_dir = Path('../test_data')

    # Load master data
    print("Loading master data...")
    facilities = load_csv(setup_dir / 'facilities.csv')
    providers = load_csv(setup_dir / 'providers.csv')

    print(f"Loaded {len(facilities)} facilities, {len(providers)} providers")

    # Generate comprehensive EDI files
    print("\nGenerating comprehensive EDI files with Phases 4-6 data elements...")
    print("Includes:")
    print("  - Phase 4: COB (Loop 2320 - SBR, CAS, OI, MOA)")
    print("  - Phase 5: Specialized claims (CR1, CR2, CR3, CR5, CR7, HSD, PWK)")
    print("  - Phase 6: Additional loops (2010BC, 2420B, MEA, HCP)")

    for facility in facilities:
        # Get providers for this facility
        facility_providers = [p for p in providers if p['facility_code'] == facility['facility_code']]

        if not facility_providers:
            print(f"Warning: No providers found for {facility['facility_code']}, skipping")
            continue

        # Generate 10,000 comprehensive claims per facility
        num_claims = 10000

        output_path = output_dir / f"claims_{facility['facility_code']}_comprehensive.edi"
        generate_edi_file(facility, facility_providers, num_claims, output_path)

    print("\nDone! Comprehensive EDI files generated in test_data/")
    print("Files include all Phases 4-6 data elements for complete 837P testing")

if __name__ == '__main__':
    main()
