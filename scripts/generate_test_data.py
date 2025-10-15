#!/usr/bin/env python3
"""
Professional SMART Test Data Generator

Generates realistic healthcare test data including:
- Organizations, Regions, Facilities
- 10,000 claims per facility with service lines and diagnoses
- Athena Health CSV format

Usage:
    python generate_test_data.py [output_dir]

Output:
    - organizations.csv
    - regions.csv
    - facilities.csv
    - claims_facility_[id].csv (10,000 claims per facility)
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import List, Dict
import uuid


class TestDataGenerator:
    """Generates realistic healthcare test data"""

    # Reference data
    FIRST_NAMES = [
        "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
        "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
        "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
        "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
        "Steven", "Kimberly", "Paul", "Emily", "Andrew", "Donna", "Joshua", "Michelle",
        "Kenneth", "Dorothy", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
        "Edward", "Deborah", "Ronald", "Stephanie", "Timothy", "Rebecca", "Jason", "Sharon",
        "Jeffrey", "Laura", "Ryan", "Cynthia", "Jacob", "Kathleen", "Gary", "Amy",
        "Nicholas", "Shirley", "Eric", "Angela", "Jonathan", "Helen", "Stephen", "Anna"
    ]

    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
        "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
        "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
        "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
        "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
        "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
        "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey"
    ]

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

    def __init__(self, output_dir: str = "./test_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.organizations = []
        self.regions = []
        self.facilities = []
        self.providers = []

    def generate_npi(self) -> str:
        """Generate a valid-looking 10-digit NPI"""
        return f"{random.randint(1000000000, 9999999999)}"

    def generate_date_range(self, start_date: datetime, days: int) -> datetime:
        """Generate a random date within range"""
        return start_date + timedelta(days=random.randint(0, days))

    def generate_organizations(self, count: int = 2):
        """Generate organizations"""
        print(f"Generating {count} organizations...")

        org_names = [
            "Regional Health System",
            "Metropolitan Medical Group",
            "Community Healthcare Network",
            "Advanced Care Alliance"
        ]

        for i in range(count):
            org = {
                "organization_id": str(uuid.uuid4()),
                "organization_code": f"ORG{i+1:03d}",
                "organization_name": org_names[i] if i < len(org_names) else f"Healthcare Org {i+1}",
                "tax_id": f"{random.randint(10, 99)}-{random.randint(1000000, 9999999)}",
                "npi": self.generate_npi(),
                "address_line1": f"{random.randint(100, 9999)} Medical Plaza",
                "city": random.choice(["New York", "Los Angeles", "Chicago", "Houston"]),
                "state_code": random.choice(["NY", "CA", "IL", "TX"]),
                "postal_code": f"{random.randint(10000, 99999)}",
                "phone": f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
                "email": f"contact@org{i+1}.com"
            }
            self.organizations.append(org)

        # Write to CSV
        filename = self.output_dir / "organizations.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=org.keys())
            writer.writeheader()
            writer.writerows(self.organizations)

        print(f"  ✓ Written to {filename}")

    def generate_regions(self, regions_per_org: int = 2):
        """Generate regions"""
        total = len(self.organizations) * regions_per_org
        print(f"Generating {total} regions ({regions_per_org} per organization)...")

        region_names = ["North", "South", "East", "West", "Central", "Metro"]

        for org in self.organizations:
            for i in range(regions_per_org):
                region = {
                    "region_id": str(uuid.uuid4()),
                    "organization_id": org["organization_id"],
                    "region_code": f"{org['organization_code']}-R{i+1}",
                    "region_name": f"{region_names[i]} Region",
                    "description": f"{region_names[i]} service area"
                }
                self.regions.append(region)

        # Write to CSV
        filename = self.output_dir / "regions.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=region.keys())
            writer.writeheader()
            writer.writerows(self.regions)

        print(f"  ✓ Written to {filename}")

    def generate_facilities(self, facilities_per_region: int = 2):
        """Generate facilities"""
        total = len(self.regions) * facilities_per_region
        print(f"Generating {total} facilities ({facilities_per_region} per region)...")

        facility_types = [
            "Medical Center", "Clinic", "Outpatient Center",
            "Urgent Care", "Surgery Center", "Specialty Clinic"
        ]

        for region in self.regions:
            for i in range(facilities_per_region):
                # Find the organization for this region
                org = next(o for o in self.organizations if o["organization_id"] == region["organization_id"])

                facility = {
                    "facility_id": str(uuid.uuid4()),
                    "organization_id": region["organization_id"],
                    "region_id": region["region_id"],
                    "facility_code": f"{region['region_code']}-F{i+1}",
                    "facility_name": f"{region['region_name']} {facility_types[i % len(facility_types)]}",
                    "npi": self.generate_npi(),
                    "tax_id": org["tax_id"],
                    "facility_type": facility_types[i % len(facility_types)],
                    "address_line1": f"{random.randint(100, 9999)} Healthcare Drive",
                    "city": org["city"],
                    "state_code": org["state_code"],
                    "postal_code": f"{random.randint(10000, 99999)}",
                    "phone": f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
                    "email": f"contact@facility{len(self.facilities)+1}.com",
                    "ehr_system": random.choice(["ATHENA", "EPIC", "CERNER"])
                }
                self.facilities.append(facility)

        # Write to CSV
        filename = self.output_dir / "facilities.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=facility.keys())
            writer.writeheader()
            writer.writerows(self.facilities)

        print(f"  ✓ Written to {filename}")

    def generate_providers(self, count: int = 50):
        """Generate providers"""
        print(f"Generating {count} providers...")

        specialties = [
            "Family Medicine", "Internal Medicine", "Pediatrics", "Cardiology",
            "Orthopedics", "Dermatology", "Neurology", "Psychiatry",
            "General Surgery", "Emergency Medicine"
        ]

        for i in range(count):
            provider = {
                "npi": self.generate_npi(),
                "last_name": random.choice(self.LAST_NAMES),
                "first_name": random.choice(self.FIRST_NAMES),
                "specialty": random.choice(specialties)
            }
            self.providers.append(provider)

        print(f"  ✓ Generated {count} providers")

    def generate_claims_for_facility(self, facility: Dict, claim_count: int = 10000):
        """Generate claims for a single facility in Athena CSV format"""

        org = next(o for o in self.organizations if o["organization_id"] == facility["organization_id"])

        print(f"  Generating {claim_count:,} claims for {facility['facility_name']}...")

        filename = self.output_dir / f"claims_{facility['facility_code']}.csv"

        # Start date: 1 year ago
        start_date = datetime.now() - timedelta(days=365)

        with open(filename, 'w', newline='') as f:
            # Athena Health CSV format
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
                patient_id = f"{facility['facility_code']}-{claim_num+1:06d}"
                last_name = random.choice(self.LAST_NAMES)
                first_name = random.choice(self.FIRST_NAMES)
                gender = random.choice(["M", "F"])

                # DOB: between 1-90 years old
                age_days = random.randint(365, 365*90)
                dob = datetime.now() - timedelta(days=age_days)

                # Service date within last year
                dos = self.generate_date_range(start_date, 365)

                # Provider
                provider = random.choice(self.providers)

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
                        "Provider NPI": provider["npi"],
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
                        "Facility Code": facility["facility_code"],
                        "Facility Name": facility["facility_name"]
                    }

                    writer.writerow(claim)

                # Progress indicator
                if (claim_num + 1) % 1000 == 0:
                    print(f"    {claim_num + 1:,} / {claim_count:,} claims generated...")

        print(f"  ✓ Written to {filename}")
        return filename

    def generate_all_claims(self, claims_per_facility: int = 10000):
        """Generate claims for all facilities"""
        total = len(self.facilities) * claims_per_facility
        print(f"\nGenerating {total:,} total claims ({claims_per_facility:,} per facility)...")

        for i, facility in enumerate(self.facilities, 1):
            print(f"\nFacility {i}/{len(self.facilities)}: {facility['facility_name']}")
            self.generate_claims_for_facility(facility, claims_per_facility)

    def generate_all(self,
                     org_count: int = 2,
                     regions_per_org: int = 2,
                     facilities_per_region: int = 2,
                     providers_count: int = 50,
                     claims_per_facility: int = 10000):
        """Generate complete test dataset"""

        print("=" * 70)
        print("Professional SMART Test Data Generator")
        print("=" * 70)
        print()

        self.generate_organizations(org_count)
        self.generate_regions(regions_per_org)
        self.generate_facilities(facilities_per_region)
        self.generate_providers(providers_count)
        self.generate_all_claims(claims_per_facility)

        print()
        print("=" * 70)
        print("Generation Complete!")
        print("=" * 70)
        print(f"Output directory: {self.output_dir.absolute()}")
        print()
        print("Summary:")
        print(f"  Organizations: {len(self.organizations)}")
        print(f"  Regions: {len(self.regions)}")
        print(f"  Facilities: {len(self.facilities)}")
        print(f"  Providers: {len(self.providers)}")
        print(f"  Claims files: {len(self.facilities)}")
        print(f"  Total claims: {len(self.facilities) * claims_per_facility:,}")
        print()


def main():
    """Main entry point"""
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "./test_data"

    generator = TestDataGenerator(output_dir)

    # Generate:
    # - 2 organizations
    # - 2 regions per organization (4 total)
    # - 2 facilities per region (8 total)
    # - 50 providers
    # - 10,000 claims per facility (80,000 total claims)

    generator.generate_all(
        org_count=2,
        regions_per_org=2,
        facilities_per_region=2,
        providers_count=50,
        claims_per_facility=10000
    )


if __name__ == "__main__":
    main()
