#!/usr/bin/env python3
"""
Regenerate claims files matching existing test_data

Reads existing:
- test_data/organizations.csv
- test_data/regions.csv
- test_data/facilities.csv
- test_data/providers.csv

Generates:
- 10,000 claims per facility in CSV format
- Matching EDI X12 837P files for each facility

Usage:
    python regenerate_claims.py
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict


class ClaimsGenerator:
    """Generates claims matching existing master data"""

    CPT_CODES = [
        ("99213", "Office Visit Level 3", 150.00),
        ("99214", "Office Visit Level 4", 200.00),
        ("99215", "Office Visit Level 5", 250.00),
        ("99203", "New Patient Level 3", 175.00),
        ("99204", "New Patient Level 4", 225.00),
        ("99205", "New Patient Level 5", 300.00),
        ("99232", "Hospital Subsequent Care", 125.00),
        ("99233", "Hospital Subsequent Care High", 175.00),
        ("99285", "Emergency Visit High", 450.00),
        ("99284", "Emergency Visit Moderate", 350.00),
        ("80053", "Comprehensive Metabolic Panel", 75.00),
        ("85025", "Complete Blood Count", 45.00),
        ("36415", "Venipuncture", 25.00),
        ("93000", "Electrocardiogram", 85.00),
        ("71045", "Chest X-Ray Single View", 120.00),
        ("71046", "Chest X-Ray 2 Views", 150.00),
        ("73070", "Elbow X-Ray", 110.00),
        ("73560", "Knee X-Ray", 125.00),
        ("76700", "Abdominal Ultrasound", 250.00),
        ("70450", "CT Head Without Contrast", 450.00),
        ("70553", "MRI Brain", 800.00),
        ("20610", "Arthrocentesis", 180.00),
        ("11042", "Debridement", 200.00),
        ("12001", "Simple Repair 2.5cm", 150.00),
        ("29125", "Forearm Splint", 100.00),
        ("96372", "Injection SubQ/IM", 35.00),
        ("90471", "Immunization Admin", 25.00),
        ("90715", "Tdap Vaccine", 55.00),
        ("J3301", "Kenalog Injection", 45.00),
        ("J1885", "Ketorolac Injection", 35.00)
    ]

    MODIFIERS = ["", "25", "59", "76", "77", "GT", "GY", "GZ", "TC", "26"]

    ICD10_CODES = [
        ("Z00.00", "Encounter for general adult exam"),
        ("I10", "Essential hypertension"),
        ("E11.9", "Type 2 diabetes without complications"),
        ("E78.5", "Hyperlipidemia"),
        ("J06.9", "Acute upper respiratory infection"),
        ("M25.511", "Pain in right shoulder"),
        ("M25.512", "Pain in left shoulder"),
        ("M54.5", "Low back pain"),
        ("R51.9", "Headache"),
        ("N39.0", "Urinary tract infection"),
        ("J44.9", "COPD unspecified"),
        ("F41.9", "Anxiety disorder"),
        ("F32.9", "Major depressive disorder"),
        ("K21.9", "GERD"),
        ("M79.3", "Panniculitis"),
        ("R10.9", "Abdominal pain"),
        ("R50.9", "Fever"),
        ("R05.9", "Cough"),
        ("R53.83", "Fatigue"),
        ("Z23", "Encounter for immunization"),
        ("M17.11", "Knee osteoarthritis right"),
        ("M19.90", "Osteoarthritis unspecified"),
        ("G43.909", "Migraine"),
        ("J02.9", "Pharyngitis"),
        ("J20.9", "Bronchitis"),
        ("B34.9", "Viral infection"),
        ("L03.90", "Cellulitis"),
        ("S93.401A", "Sprain ankle right initial"),
        ("S83.201A", "Meniscus tear right initial"),
        ("T14.90XA", "Injury unspecified initial")
    ]

    PAYERS = [
        ("BCBS001", "Blue Cross Blue Shield"),
        ("AETNA01", "Aetna"),
        ("UHC0001", "UnitedHealthcare"),
        ("CIGNA01", "Cigna"),
        ("HUMN001", "Humana"),
        ("MCARE01", "Medicare"),
        ("MCAID01", "Medicaid"),
        ("ANTM001", "Anthem"),
        ("CENT001", "Centene"),
        ("MOLINA1", "Molina Healthcare")
    ]

    PLACE_OF_SERVICE = ["11", "22", "23", "31", "81"]  # Office, Outpatient, ER, SNF, Lab

    FIRST_NAMES = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa"
    ]

    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
        "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White"
    ]

    def __init__(self, test_data_dir: str = "./test_data"):
        self.test_data_dir = Path(test_data_dir)
        self.organizations = []
        self.regions = []
        self.facilities = []
        self.providers = []
        self.providers_by_facility = {}

    def load_master_data(self):
        """Load existing master data from test_data"""
        print("Loading master data...")

        # Load organizations
        with open(self.test_data_dir / "organizations.csv", 'r') as f:
            reader = csv.DictReader(f)
            self.organizations = list(reader)
        print(f"  Loaded {len(self.organizations)} organizations")

        # Load regions
        with open(self.test_data_dir / "regions.csv", 'r') as f:
            reader = csv.DictReader(f)
            self.regions = list(reader)
        print(f"  Loaded {len(self.regions)} regions")

        # Load facilities
        with open(self.test_data_dir / "facilities.csv", 'r') as f:
            reader = csv.DictReader(f)
            self.facilities = list(reader)
        print(f"  Loaded {len(self.facilities)} facilities")

        # Load providers
        with open(self.test_data_dir / "providers.csv", 'r') as f:
            reader = csv.DictReader(f)
            self.providers = list(reader)
        print(f"  Loaded {len(self.providers)} providers")

        # Group providers by facility
        for provider in self.providers:
            facility_code = provider['facility_code']
            if facility_code not in self.providers_by_facility:
                self.providers_by_facility[facility_code] = []
            self.providers_by_facility[facility_code].append(provider)

        print(f"  Grouped providers across {len(self.providers_by_facility)} facilities")
        print()

    def generate_date_range(self, start_date: datetime, days: int) -> datetime:
        """Generate a random date within range"""
        return start_date + timedelta(days=random.randint(0, days))

    def generate_claims_csv(self, facility: Dict, org: Dict, region: Dict, claim_count: int = 10000):
        """Generate claims CSV for a facility"""

        facility_code = facility['facility_code']
        org_code = facility['organization_code']
        region_code = facility['region_code']

        filename = self.test_data_dir / f"claims_{org_code}-{region_code}-{facility_code}.csv"

        print(f"  Generating {claim_count:,} claims for {facility['facility_name']}...")
        print(f"    Output: {filename.name}")

        # Get providers for this facility
        facility_providers = self.providers_by_facility.get(facility_code, [])
        if not facility_providers:
            print(f"    WARNING: No providers found for {facility_code}, using all providers")
            facility_providers = self.providers

        # Start date: 1 year ago
        start_date = datetime.now() - timedelta(days=365)

        with open(filename, 'w', newline='') as f:
            fieldnames = [
                "Patient ID", "DOS", "Provider NPI", "CPT", "Modifier 1", "Modifier 2",
                "Units", "Charges", "Diagnosis 1", "Diagnosis 2", "Diagnosis 3", "Diagnosis 4",
                "Patient Last Name", "Patient First Name", "DOB", "Gender", "POS",
                "Payer ID", "Payer Name", "Member ID", "Facility Code", "Facility Name"
            ]

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for claim_num in range(claim_count):
                # Generate patient demographics
                patient_id = f"{facility_code}-P{claim_num+1:06d}"
                last_name = random.choice(self.LAST_NAMES)
                first_name = random.choice(self.FIRST_NAMES)
                gender = random.choice(["M", "F"])

                # DOB: between 1-90 years old
                age_days = random.randint(365, 365*90)
                dob = datetime.now() - timedelta(days=age_days)

                # Service date within last year
                dos = self.generate_date_range(start_date, 365)

                # Provider from this facility
                provider = random.choice(facility_providers)

                # Payer
                payer_id, payer_name = random.choice(self.PAYERS)

                # Member ID
                member_id = f"MEM{random.randint(100000000, 999999999)}"

                # Place of service
                pos = random.choice(self.PLACE_OF_SERVICE)

                # Number of service lines (1-5)
                num_lines = random.choices([1, 2, 3, 4, 5], weights=[50, 30, 15, 4, 1])[0]

                # Number of diagnoses (1-4)
                num_dx = random.choices([1, 2, 3, 4], weights=[30, 40, 20, 10])[0]
                diagnoses = random.sample(self.ICD10_CODES, num_dx)

                # Generate service lines
                for line_num in range(num_lines):
                    cpt_code, cpt_desc, base_charge = random.choice(self.CPT_CODES)

                    # Vary charge by +/- 20%
                    charge = base_charge * random.uniform(0.8, 1.2)

                    # Units (usually 1, sometimes 2-3)
                    units = random.choices([1, 2, 3], weights=[85, 10, 5])[0]

                    # Modifiers
                    mod1 = random.choice(self.MODIFIERS)
                    mod2 = random.choice(["", ""] + self.MODIFIERS)  # Less common

                    claim = {
                        "Patient ID": patient_id,
                        "DOS": dos.strftime("%Y-%m-%d"),
                        "Provider NPI": provider["provider_npi"],
                        "CPT": cpt_code,
                        "Modifier 1": mod1,
                        "Modifier 2": mod2,
                        "Units": units,
                        "Charges": f"{charge:.2f}",
                        "Diagnosis 1": diagnoses[0][0] if len(diagnoses) > 0 else "",
                        "Diagnosis 2": diagnoses[1][0] if len(diagnoses) > 1 else "",
                        "Diagnosis 3": diagnoses[2][0] if len(diagnoses) > 2 else "",
                        "Diagnosis 4": diagnoses[3][0] if len(diagnoses) > 3 else "",
                        "Patient Last Name": last_name,
                        "Patient First Name": first_name,
                        "DOB": dob.strftime("%Y-%m-%d"),
                        "Gender": gender,
                        "POS": pos,
                        "Payer ID": payer_id,
                        "Payer Name": payer_name,
                        "Member ID": member_id,
                        "Facility Code": facility_code,
                        "Facility Name": facility["facility_name"]
                    }

                    writer.writerow(claim)

                # Progress indicator
                if (claim_num + 1) % 2000 == 0:
                    print(f"      {claim_num + 1:,} / {claim_count:,} claims...")

        print(f"    Complete: {filename.name}")
        return filename

    def generate_all_claims(self, claims_per_facility: int = 10000):
        """Generate claims for all facilities"""
        total = len(self.facilities) * claims_per_facility
        print(f"Generating {total:,} total claims ({claims_per_facility:,} per facility)...\n")

        for i, facility in enumerate(self.facilities, 1):
            print(f"Facility {i}/{len(self.facilities)}:")

            # Find org and region for this facility
            org = next((o for o in self.organizations if o['organization_code'] == facility['organization_code']), None)
            region = next((r for r in self.regions if r['organization_code'] == facility['organization_code']
                          and r['region_code'] == facility['region_code']), None)

            self.generate_claims_csv(facility, org, region, claims_per_facility)
            print()

    def run(self, claims_per_facility: int = 10000):
        """Main execution"""
        print("=" * 70)
        print("Claims Data Regeneration")
        print("=" * 70)
        print()

        self.load_master_data()
        self.generate_all_claims(claims_per_facility)

        print("=" * 70)
        print("Generation Complete!")
        print("=" * 70)
        print(f"Output directory: {self.test_data_dir.absolute()}")
        print()
        print("Summary:")
        print(f"  Facilities: {len(self.facilities)}")
        print(f"  Providers: {len(self.providers)}")
        print(f"  Claims files: {len(self.facilities)}")
        print(f"  Total claims: {len(self.facilities) * claims_per_facility:,}")
        print()


def main():
    generator = ClaimsGenerator("./test_data")
    generator.run(claims_per_facility=10000)


if __name__ == "__main__":
    main()
