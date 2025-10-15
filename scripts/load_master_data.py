#!/usr/bin/env python3
"""
Load Organization/Region/Facility Master Data

Loads organization hierarchy from CSV files into the Professional SMART database.

Usage:
    python load_master_data.py [data_directory]

Environment Variables:
    DATABASE_URL - PostgreSQL connection string
    Or set in .env file

Example:
    python load_master_data.py test_data
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime


class MasterDataLoader:
    """Loads organization, region, and facility data from CSV files"""

    def __init__(self, data_dir: str, database_url: str = None):
        self.data_dir = Path(data_dir)
        self.database_url = database_url or os.getenv('DATABASE_URL')

        if not self.database_url:
            raise ValueError("DATABASE_URL not found. Set in environment or .env file.")

        self.conn = None
        self.organizations_loaded = 0
        self.regions_loaded = 0
        self.facilities_loaded = 0

    def connect(self):
        """Connect to PostgreSQL database"""
        print(f"Connecting to database...")
        try:
            self.conn = psycopg2.connect(self.database_url)
            print("  ✓ Connected successfully")
        except Exception as e:
            print(f"  ✗ Connection failed: {e}")
            raise

    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("Database connection closed")

    def load_organizations(self) -> Dict[str, str]:
        """
        Load organizations from CSV
        Returns mapping of organization_id from CSV to database UUID
        """
        csv_file = self.data_dir / "organizations.csv"

        if not csv_file.exists():
            print(f"  ⚠ organizations.csv not found in {self.data_dir}")
            return {}

        print(f"\nLoading organizations from {csv_file}...")

        org_id_map = {}  # Maps CSV UUID to database UUID

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if not rows:
                print("  ⚠ No organizations to load")
                return {}

            print(f"  Found {len(rows)} organizations")

            cursor = self.conn.cursor()

            for row in rows:
                try:
                    # Insert organization
                    cursor.execute("""
                        INSERT INTO claims.organization (
                            organization_code, organization_name, tax_id, npi,
                            address_line1, address_line2, city, state_code,
                            postal_code, country_code, phone, email,
                            is_active, created_by
                        ) VALUES (
                            %(organization_code)s, %(organization_name)s, %(tax_id)s, %(npi)s,
                            %(address_line1)s, %(address_line2)s, %(city)s, %(state_code)s,
                            %(postal_code)s, 'USA', %(phone)s, %(email)s,
                            true, 'system'
                        )
                        ON CONFLICT (organization_code)
                        DO UPDATE SET
                            organization_name = EXCLUDED.organization_name,
                            updated_at = CURRENT_TIMESTAMP,
                            updated_by = 'system'
                        RETURNING organization_id
                    """, row)

                    db_org_id = cursor.fetchone()[0]
                    org_id_map[row['organization_id']] = str(db_org_id)

                    print(f"  ✓ Loaded: {row['organization_code']} - {row['organization_name']}")
                    self.organizations_loaded += 1

                except Exception as e:
                    print(f"  ✗ Error loading {row.get('organization_code')}: {e}")
                    self.conn.rollback()
                    raise

            self.conn.commit()
            print(f"  ✓ Successfully loaded {self.organizations_loaded} organizations")

        return org_id_map

    def load_regions(self, org_id_map: Dict[str, str]) -> Dict[str, str]:
        """
        Load regions from CSV
        Returns mapping of region_id from CSV to database UUID
        """
        csv_file = self.data_dir / "regions.csv"

        if not csv_file.exists():
            print(f"  ⚠ regions.csv not found in {self.data_dir}")
            return {}

        print(f"\nLoading regions from {csv_file}...")

        region_id_map = {}  # Maps CSV UUID to database UUID

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if not rows:
                print("  ⚠ No regions to load")
                return {}

            print(f"  Found {len(rows)} regions")

            cursor = self.conn.cursor()

            for row in rows:
                try:
                    # Map CSV organization_id to database UUID
                    csv_org_id = row['organization_id']
                    db_org_id = org_id_map.get(csv_org_id)

                    if not db_org_id:
                        print(f"  ✗ Organization ID {csv_org_id} not found for region {row['region_code']}")
                        continue

                    # Insert region
                    cursor.execute("""
                        INSERT INTO claims.region (
                            organization_id, region_code, region_name, description,
                            is_active, created_by
                        ) VALUES (
                            %(organization_id)s, %(region_code)s, %(region_name)s, %(description)s,
                            true, 'system'
                        )
                        ON CONFLICT (organization_id, region_code)
                        DO UPDATE SET
                            region_name = EXCLUDED.region_name,
                            description = EXCLUDED.description,
                            updated_at = CURRENT_TIMESTAMP,
                            updated_by = 'system'
                        RETURNING region_id
                    """, {
                        'organization_id': db_org_id,
                        'region_code': row['region_code'],
                        'region_name': row['region_name'],
                        'description': row.get('description', '')
                    })

                    db_region_id = cursor.fetchone()[0]
                    region_id_map[row['region_id']] = str(db_region_id)

                    print(f"  ✓ Loaded: {row['region_code']} - {row['region_name']}")
                    self.regions_loaded += 1

                except Exception as e:
                    print(f"  ✗ Error loading {row.get('region_code')}: {e}")
                    self.conn.rollback()
                    raise

            self.conn.commit()
            print(f"  ✓ Successfully loaded {self.regions_loaded} regions")

        return region_id_map

    def load_facilities(self, org_id_map: Dict[str, str], region_id_map: Dict[str, str]) -> Dict[str, str]:
        """
        Load facilities from CSV
        Returns mapping of facility_id from CSV to database UUID
        """
        csv_file = self.data_dir / "facilities.csv"

        if not csv_file.exists():
            print(f"  ⚠ facilities.csv not found in {self.data_dir}")
            return {}

        print(f"\nLoading facilities from {csv_file}...")

        facility_id_map = {}  # Maps CSV UUID to database UUID

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            if not rows:
                print("  ⚠ No facilities to load")
                return {}

            print(f"  Found {len(rows)} facilities")

            cursor = self.conn.cursor()

            for row in rows:
                try:
                    # Map CSV organization_id to database UUID
                    csv_org_id = row['organization_id']
                    db_org_id = org_id_map.get(csv_org_id)

                    if not db_org_id:
                        print(f"  ✗ Organization ID {csv_org_id} not found for facility {row['facility_code']}")
                        continue

                    # Map CSV region_id to database UUID (if exists)
                    csv_region_id = row.get('region_id')
                    db_region_id = region_id_map.get(csv_region_id) if csv_region_id else None

                    # Insert facility
                    cursor.execute("""
                        INSERT INTO claims.facility (
                            organization_id, region_id, facility_code, facility_name,
                            npi, tax_id, facility_type,
                            address_line1, address_line2, city, state_code,
                            postal_code, country_code, phone, email, ehr_system,
                            is_active, created_by
                        ) VALUES (
                            %(organization_id)s, %(region_id)s, %(facility_code)s, %(facility_name)s,
                            %(npi)s, %(tax_id)s, %(facility_type)s,
                            %(address_line1)s, %(address_line2)s, %(city)s, %(state_code)s,
                            %(postal_code)s, 'USA', %(phone)s, %(email)s, %(ehr_system)s,
                            true, 'system'
                        )
                        ON CONFLICT (organization_id, facility_code)
                        DO UPDATE SET
                            facility_name = EXCLUDED.facility_name,
                            region_id = EXCLUDED.region_id,
                            npi = EXCLUDED.npi,
                            ehr_system = EXCLUDED.ehr_system,
                            updated_at = CURRENT_TIMESTAMP,
                            updated_by = 'system'
                        RETURNING facility_id
                    """, {
                        'organization_id': db_org_id,
                        'region_id': db_region_id,
                        'facility_code': row['facility_code'],
                        'facility_name': row['facility_name'],
                        'npi': row.get('npi'),
                        'tax_id': row.get('tax_id'),
                        'facility_type': row.get('facility_type'),
                        'address_line1': row.get('address_line1'),
                        'address_line2': row.get('address_line2'),
                        'city': row.get('city'),
                        'state_code': row.get('state_code'),
                        'postal_code': row.get('postal_code'),
                        'phone': row.get('phone'),
                        'email': row.get('email'),
                        'ehr_system': row.get('ehr_system')
                    })

                    db_facility_id = cursor.fetchone()[0]
                    facility_id_map[row['facility_id']] = str(db_facility_id)

                    print(f"  ✓ Loaded: {row['facility_code']} - {row['facility_name']}")
                    self.facilities_loaded += 1

                except Exception as e:
                    print(f"  ✗ Error loading {row.get('facility_code')}: {e}")
                    self.conn.rollback()
                    raise

            self.conn.commit()
            print(f"  ✓ Successfully loaded {self.facilities_loaded} facilities")

        return facility_id_map

    def verify_data(self):
        """Verify loaded data"""
        print("\nVerifying loaded data...")

        cursor = self.conn.cursor()

        # Count organizations
        cursor.execute("SELECT COUNT(*) FROM claims.organization WHERE is_active = true")
        org_count = cursor.fetchone()[0]
        print(f"  Organizations in database: {org_count}")

        # Count regions
        cursor.execute("SELECT COUNT(*) FROM claims.region WHERE is_active = true")
        region_count = cursor.fetchone()[0]
        print(f"  Regions in database: {region_count}")

        # Count facilities
        cursor.execute("SELECT COUNT(*) FROM claims.facility WHERE is_active = true")
        facility_count = cursor.fetchone()[0]
        print(f"  Facilities in database: {facility_count}")

        # Show hierarchy
        print("\nOrganization Hierarchy:")
        cursor.execute("""
            SELECT
                o.organization_code,
                o.organization_name,
                COUNT(DISTINCT r.region_id) as region_count,
                COUNT(DISTINCT f.facility_id) as facility_count
            FROM claims.organization o
            LEFT JOIN claims.region r ON o.organization_id = r.organization_id
            LEFT JOIN claims.facility f ON o.organization_id = f.organization_id
            WHERE o.is_active = true
            GROUP BY o.organization_id, o.organization_code, o.organization_name
            ORDER BY o.organization_code
        """)

        for row in cursor.fetchall():
            org_code, org_name, regions, facilities = row
            print(f"  {org_code}: {org_name}")
            print(f"    - {regions} regions")
            print(f"    - {facilities} facilities")

    def load_all(self):
        """Load all master data"""
        print("=" * 70)
        print("Professional SMART Master Data Loader")
        print("=" * 70)
        print(f"Data directory: {self.data_dir.absolute()}")
        print()

        try:
            self.connect()

            # Load in order: organizations -> regions -> facilities
            org_id_map = self.load_organizations()
            region_id_map = self.load_regions(org_id_map)
            facility_id_map = self.load_facilities(org_id_map, region_id_map)

            self.verify_data()

            print()
            print("=" * 70)
            print("Loading Complete!")
            print("=" * 70)
            print(f"Summary:")
            print(f"  Organizations loaded: {self.organizations_loaded}")
            print(f"  Regions loaded: {self.regions_loaded}")
            print(f"  Facilities loaded: {self.facilities_loaded}")
            print()
            print("Next steps:")
            print("  1. Verify data: SELECT * FROM claims.organization;")
            print("  2. Load claims: Copy test_data/claims_*.csv to input directory")
            print("  3. Start service: professional-smart console")
            print()

        except Exception as e:
            print(f"\n✗ Error: {e}")
            if self.conn:
                self.conn.rollback()
            raise
        finally:
            self.disconnect()


def load_env_file():
    """Load .env file if it exists"""
    env_file = Path('.env')
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value


def main():
    """Main entry point"""
    # Load .env file
    load_env_file()

    # Get data directory from command line or use default
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "test_data"

    # Check if directory exists
    if not Path(data_dir).exists():
        print(f"Error: Directory '{data_dir}' not found")
        print()
        print("Usage: python load_master_data.py [data_directory]")
        print()
        print("Example: python load_master_data.py test_data")
        sys.exit(1)

    # Load data
    loader = MasterDataLoader(data_dir)
    loader.load_all()


if __name__ == "__main__":
    main()
