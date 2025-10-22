#!/usr/bin/env python3
"""
Generate EDI X12 837P files from CSV claims

Reads CSV claims files and generates corresponding EDI X12 837P files
for professional claims submission.

Usage:
    python generate_edi_from_csv.py
"""

import csv
from datetime import datetime
from pathlib import Path
from collections import defaultdict


class EDIGenerator:
    """Generates EDI X12 837P files from CSV claims"""

    def __init__(self, test_data_dir: str = "./test_data"):
        self.test_data_dir = Path(test_data_dir)
        self.control_number = 1
        self.transaction_set_control_number = 1

    def format_date(self, date_str: str) -> str:
        """Convert YYYY-MM-DD to YYYYMMDD"""
        return date_str.replace("-", "")

    def format_amount(self, amount: str) -> str:
        """Format dollar amount for EDI (no decimal point)"""
        # Remove dollar sign if present and convert to cents
        amount = amount.replace("$", "").strip()
        dollars, cents = amount.split(".")
        return f"{int(dollars)}{cents}"

    def generate_isa_segment(self, sender_id: str, receiver_id: str, timestamp: datetime) -> str:
        """Generate ISA (Interchange Control Header)"""
        date = timestamp.strftime("%y%m%d")
        time = timestamp.strftime("%H%M")
        control_num = f"{self.control_number:09d}"

        return (
            f"ISA*00*          *00*          *ZZ*{sender_id:<15}*"
            f"ZZ*{receiver_id:<15}*{date}*{time}*^*00501*{control_num}*0*P*:~"
        )

    def generate_gs_segment(self, sender_code: str, receiver_code: str, timestamp: datetime) -> str:
        """Generate GS (Functional Group Header)"""
        date = timestamp.strftime("%Y%m%d")
        time = timestamp.strftime("%H%M")
        control_num = f"{self.control_number}"

        return (
            f"GS*HC*{sender_code}*{receiver_code}*{date}*{time}*{control_num}*X*005010X222A1~"
        )

    def generate_st_segment(self) -> str:
        """Generate ST (Transaction Set Header)"""
        control_num = f"{self.transaction_set_control_number:04d}"
        return f"ST*837*{control_num}*005010X222A1~"

    def generate_bht_segment(self, timestamp: datetime) -> str:
        """Generate BHT (Beginning of Hierarchical Transaction)"""
        date = timestamp.strftime("%Y%m%d")
        time = timestamp.strftime("%H%M")
        return f"BHT*0019*00*{self.transaction_set_control_number}*{date}*{time}*CH~"

    def generate_nm1_segment(self, entity_type: str, entity_code: str, name: str, id_qualifier: str = "", id_code: str = "") -> str:
        """Generate NM1 (Individual or Organizational Name)"""
        if id_qualifier and id_code:
            return f"NM1*{entity_type}*{entity_code}*{name}****{id_qualifier}*{id_code}~"
        return f"NM1*{entity_type}*{entity_code}*{name}~"

    def process_csv_to_edi(self, csv_filename: Path, facility_code: str):
        """Convert CSV claims to EDI X12 837P format"""

        edi_filename = csv_filename.with_suffix('.edi')
        print(f"  Converting {csv_filename.name} -> {edi_filename.name}")

        # Read all claims from CSV
        claims_by_patient = defaultdict(list)

        with open(csv_filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = row['Patient Control Number']
                claims_by_patient[patient_id].append(row)

        # Group claims into encounters (same patient, same DOS)
        encounters = defaultdict(list)
        for patient_id, claims in claims_by_patient.items():
            for claim in claims:
                encounter_key = (patient_id, claim['Date of Service'])
                encounters[encounter_key].append(claim)

        print(f"    Processing {len(encounters)} encounters from {len(claims_by_patient)} patients")

        # Generate EDI file
        timestamp = datetime.now()

        with open(edi_filename, 'w') as f:
            # ISA - Interchange Control Header
            f.write(self.generate_isa_segment(facility_code, "PAYER001", timestamp))
            f.write("\n")

            # GS - Functional Group Header
            f.write(self.generate_gs_segment(facility_code, "PAYER001", timestamp))
            f.write("\n")

            # ST - Transaction Set Header
            f.write(self.generate_st_segment())
            f.write("\n")

            # BHT - Beginning of Hierarchical Transaction
            f.write(self.generate_bht_segment(timestamp))
            f.write("\n")

            # NM1 - Submitter Name
            first_claim = next(iter(encounters.values()))[0]
            f.write(f"NM1*41*2*{first_claim['Facility Name']}****46*{facility_code}~\n")

            # PER - Submitter Contact
            f.write("PER*IC*Contact Name*TE*5555551234~\n")

            # NM1 - Receiver Name
            f.write(f"NM1*40*2*PAYER NAME****46*PAYER001~\n")

            # HL - Hierarchical Level (Billing Provider)
            hl_count = 1
            f.write(f"HL*{hl_count}**20*1~\n")

            # NM1 - Billing Provider
            f.write(f"NM1*85*2*{first_claim['Facility Name']}****XX*{first_claim['Facility Code']}~\n")

            # N3/N4 - Address would go here
            f.write(f"N3*{first_claim['Facility Name']} Address~\n")
            f.write("N4*City*ST*12345~\n")

            # REF - Billing Provider Tax ID
            f.write(f"REF*EI*123456789~\n")

            encounter_num = 0
            for (patient_id, dos), service_lines in encounters.items():
                encounter_num += 1

                first_line = service_lines[0]

                # HL - Subscriber Level
                hl_count += 1
                f.write(f"HL*{hl_count}*1*22*0~\n")

                # SBR - Subscriber Information
                f.write("SBR*P*18******CI~\n")

                # NM1 - Subscriber Name
                f.write(f"NM1*IL*1*{first_line['Patient Last Name']}*{first_line['Patient First Name']}****MI*{first_line['Member ID']}~\n")

                # N3/N4 - Subscriber Address
                f.write("N3*123 Patient St~\n")
                f.write("N4*City*ST*12345~\n")

                # DMG - Subscriber Demographics
                dob = self.format_date(first_line['DOB'])
                f.write(f"DMG*D8*{dob}*{first_line['Gender']}~\n")

                # NM1 - Payer Name
                f.write(f"NM1*PR*2*{first_line['Payer Name']}****PI*{first_line['Payer ID']}~\n")

                # CLM - Claim Information
                total_charges = sum(float(line['Charge Amount']) for line in service_lines)
                f.write(f"CLM*{patient_id}*{total_charges:.2f}***11:B:1*Y*A*Y*Y~\n")

                # DTP - Service Date
                dos_formatted = self.format_date(dos)
                f.write(f"DTP*472*D8*{dos_formatted}~\n")

                # REF - Claim Reference
                f.write(f"REF*D9*{patient_id}~\n")

                # HI - Diagnosis Codes
                diagnoses = []
                for i in range(1, 5):
                    dx_field = f'Diagnosis {i}'
                    if first_line[dx_field]:
                        qualifier = "ABK" if i == 1 else "ABF"
                        diagnoses.append(f"{qualifier}:{first_line[dx_field]}")

                if diagnoses:
                    f.write(f"HI*{':'.join(diagnoses)}~\n")

                # NM1 - Rendering Provider
                f.write(f"NM1*82*1*Provider*Name****XX*{first_line['Provider NPI']}~\n")

                # Service Lines
                for line_num, service_line in enumerate(service_lines, 1):
                    # LX - Service Line Number
                    f.write(f"LX*{line_num}~\n")

                    # SV1 - Professional Service
                    cpt = service_line['Procedure Code']
                    mod1 = service_line['Modifier 1']
                    mod2 = service_line['Modifier 2']

                    mods = ""
                    if mod1:
                        mods = f":{mod1}"
                    if mod2:
                        mods += f":{mod2}"

                    charges = service_line['Charge Amount']
                    units = service_line['Units']
                    pos = service_line['Place of Service']

                    f.write(f"SV1*HC:{cpt}{mods}*{charges}*UN*{units}***{pos}~\n")

                    # DTP - Service Date
                    f.write(f"DTP*472*D8*{dos_formatted}~\n")

            # SE - Transaction Set Trailer
            # Count segments (simplified - in real implementation would count all segments)
            segment_count = 100  # Placeholder
            f.write(f"SE*{segment_count}*{self.transaction_set_control_number:04d}~\n")

            # GE - Functional Group Trailer
            f.write(f"GE*1*{self.control_number}~\n")

            # IEA - Interchange Control Trailer
            f.write(f"IEA*1*{self.control_number:09d}~\n")

        self.transaction_set_control_number += 1
        self.control_number += 1

        print(f"    Complete: {edi_filename.name} ({encounter_num} encounters)")
        return edi_filename

    def generate_all_edi(self):
        """Generate EDI files for all CSV claims files"""
        csv_files = list(self.test_data_dir.glob("claims_*.csv"))

        print(f"Found {len(csv_files)} claims CSV files\n")

        for csv_file in csv_files:
            # Extract facility code from filename: claims_ORG001-R1-F1.csv -> ORG001-R1-F1
            facility_code = csv_file.stem.replace("claims_", "")

            self.process_csv_to_edi(csv_file, facility_code)
            print()

    def run(self):
        """Main execution"""
        print("=" * 70)
        print("EDI X12 837P Generation from CSV")
        print("=" * 70)
        print()

        self.generate_all_edi()

        print("=" * 70)
        print("Generation Complete!")
        print("=" * 70)
        print(f"Output directory: {self.test_data_dir.absolute()}")
        print()


def main():
    generator = EDIGenerator("../test_data")
    generator.run()


if __name__ == "__main__":
    main()
