#!/usr/bin/env python3
"""
Generate test EDI 837p files from master data CSV files.
Creates realistic EDI files with Loop 2310C (Service Facility) containing facility NPIs.
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# Read master data
def load_csv(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

# Sample diagnoses (ICD-10)
DIAGNOSES = [
    'E78.5',  # Hyperlipidemia
    'I10',    # Essential hypertension
    'E11.9',  # Type 2 diabetes
    'J06.9',  # Upper respiratory infection
    'M79.3',  # Myalgia
    'R10.9',  # Abdominal pain
    'F41.9',  # Anxiety disorder
    'M54.5',  # Low back pain
    'K21.9',  # GERD
    'Z23',    # Immunization
]

# Sample procedures (CPT codes)
PROCEDURES = [
    ('99213', 'Office visit established', 85.00),
    ('99214', 'Office visit detailed', 125.00),
    ('99215', 'Office visit comprehensive', 170.00),
    ('36415', 'Venipuncture', 22.00),
    ('85025', 'Complete blood count', 45.00),
    ('80053', 'Comprehensive metabolic panel', 55.00),
    ('93000', 'ECG', 65.00),
    ('71046', 'Chest X-ray', 95.00),
]

# Payers
PAYERS = [
    ('HUMN001', 'Humana'),
    ('UHC0001', 'UnitedHealthcare'),
    ('AETNA01', 'Aetna'),
    ('CIGNA01', 'Cigna'),
    ('BCBS001', 'Blue Cross Blue Shield'),
]

# Patient names
PATIENT_NAMES = [
    ('Smith', 'John', 'M', '19750315'),
    ('Johnson', 'Mary', 'F', '19820622'),
    ('Williams', 'Robert', 'M', '19691204'),
    ('Brown', 'Patricia', 'F', '19580918'),
    ('Jones', 'Michael', 'M', '19920503'),
    ('Garcia', 'Jennifer', 'F', '19880827'),
    ('Miller', 'David', 'M', '19630711'),
    ('Davis', 'Linda', 'F', '19750129'),
    ('Rodriguez', 'James', 'M', '19911015'),
    ('Martinez', 'Barbara', 'F', '19670408'),
]

def generate_isa_segment(sender_id, receiver_id, control_num, date_str, time_str):
    """Generate ISA (Interchange Control Header) segment"""
    return f"ISA*00*          *00*          *ZZ*{sender_id:<15}*ZZ*{receiver_id:<15}*{date_str}*{time_str}*^*00501*{control_num:>09}*0*P*:~"

def generate_gs_segment(sender_id, receiver_id, date_str, time_str, group_num):
    """Generate GS (Functional Group Header) segment"""
    return f"GS*HC*{sender_id}*{receiver_id}*{date_str}*{time_str}*{group_num}*X*005010X222A1~"

def generate_claim(claim_num, facility, provider, patient_idx, dos_date):
    """Generate a single claim with Loop 2310C (Service Facility)"""
    last_name, first_name, gender, dob = PATIENT_NAMES[patient_idx % len(PATIENT_NAMES)]
    member_id = f"MEM{random.randint(100000000, 999999999)}"
    payer_code, payer_name = random.choice(PAYERS)

    # Random diagnosis and procedure
    diagnosis = random.choice(DIAGNOSES)
    procedure_code, procedure_desc, charge = random.choice(PROCEDURES)

    claim_lines = []

    # Subscriber (patient) level
    claim_lines.append(f"HL*{claim_num}*1*22*0~")
    claim_lines.append(f"SBR*P*18******CI~")

    # Subscriber name (NM1*IL - Insured/Patient)
    claim_lines.append(f"NM1*IL*1*{last_name}*{first_name}****MI*{member_id}~")
    claim_lines.append(f"N3*123 Patient St~")
    claim_lines.append(f"N4*Chicago*IL*{random.randint(60000, 60999)}~")
    claim_lines.append(f"DMG*D8*{dob}*{gender}~")

    # Payer name (NM1*PR)
    claim_lines.append(f"NM1*PR*2*{payer_name}****PI*{payer_code}~")

    # Claim information
    claim_id = f"{facility['facility_code']}-{claim_num:06d}"
    claim_lines.append(f"CLM*{claim_id}*{charge:.2f}***11:B:1*Y*A*Y*Y~")
    claim_lines.append(f"DTP*472*D8*{dos_date}~")
    claim_lines.append(f"REF*D9*{claim_id}~")
    claim_lines.append(f"HI*ABK:{diagnosis}~")

    # Rendering provider (NM1*82)
    claim_lines.append(f"NM1*82*1*{provider['last_name']}*{provider['first_name']}****XX*{provider['provider_npi']}~")

    # **CRITICAL: Loop 2310C - Service Facility Location (with NPI)**
    claim_lines.append(f"NM1*77*2*{facility['facility_name']}****XX*{facility['npi']}~")
    claim_lines.append(f"N3*{facility.get('address_line1', 'Medical Center Address')}~")
    claim_lines.append(f"N4*{facility.get('city', 'Chicago')}*{facility.get('state_code', 'IL')}*{facility.get('postal_code', '60601')}~")

    # Service line
    claim_lines.append(f"LX*1~")
    claim_lines.append(f"SV1*HC:{procedure_code}*{charge:.2f}*UN*1***1~")
    claim_lines.append(f"DTP*472*D8*{dos_date}~")

    return claim_lines

def generate_edi_file(facility, providers, num_claims, output_path):
    """Generate complete EDI 837p file for a facility"""
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

    # ST segment (Transaction Set Header)
    lines.append("ST*837*0001*005010X222A1~")

    # BHT segment (Beginning of Hierarchical Transaction)
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

    # Generate claims
    for claim_idx in range(num_claims):
        provider = providers[claim_idx % len(providers)]

        # Random date of service in last 6 months
        days_ago = random.randint(1, 180)
        dos_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y%m%d')

        claim_lines = generate_claim(claim_idx + 2, facility, provider, claim_idx, dos_date)
        lines.extend(claim_lines)

    # SE segment (Transaction Set Trailer)
    lines.append(f"SE*{len(lines) + 1}*0001~")

    # GE segment (Functional Group Trailer)
    lines.append("GE*1*1~")

    # IEA segment (Interchange Control Trailer)
    lines.append(f"IEA*1*{control_num:>09}~")

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Generated: {output_path}")
    print(f"  Facility: {facility['facility_code']} ({facility['facility_name']})")
    print(f"  NPI: {facility['npi']}")
    print(f"  Claims: {num_claims}")

def main():
    # Paths
    setup_dir = Path('../test_data/setup')
    output_dir = Path('../test_data')

    # Load master data
    print("Loading master data...")
    facilities = load_csv(setup_dir / 'facilities.csv')
    providers = load_csv(setup_dir / 'providers.csv')

    print(f"Loaded {len(facilities)} facilities, {len(providers)} providers")

    # Generate EDI files for each facility
    print("\nGenerating EDI files...")

    for facility in facilities:
        # Get providers for this facility
        facility_providers = [p for p in providers if p['facility_code'] == facility['facility_code']]

        if not facility_providers:
            print(f"Warning: No providers found for {facility['facility_code']}, skipping")
            continue

        # Generate 5-10 claims per facility
        num_claims = random.randint(5, 10)

        output_path = output_dir / f"claims_{facility['facility_code']}.edi"
        generate_edi_file(facility, facility_providers, num_claims, output_path)

    print("\nDone! EDI files generated in test_data/")

if __name__ == '__main__':
    main()
