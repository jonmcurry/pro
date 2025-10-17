# Professional SMART Master Data Loader GUI

Graphical user interface for loading master data (organizations, regions, facilities, providers) into the Professional SMART database from CSV files.

## Features

- **Intuitive File Selection**: Browse for individual CSV files or load an entire directory at once
- **Database Connection Status**: Real-time connection status indicator
- **Data Validation**: Validates all CSV data before import with detailed error messages
- **Automatic Import**: Automatically proceeds to import after successful validation
- **Progress Tracking**: Real-time progress display during import
- **Detailed Logging**: Timestamped log panel showing all operations and results
- **Template Generation**: Generate CSV template files with example data
- **Results Summary**: Complete summary of imported records

## Usage

### Accessing the Application

After installation, the GUI can be launched from:
- **Start Menu**: Professional SMART → Load Master Data
- **Command Line**: `C:\Program Files\Professional SMART\bin\pro-data-loader-gui.exe`

### Step-by-Step Guide

#### 1. Check Database Connection
When the application opens, check the top-right corner for the connection status:
- **Green dot**: Connected successfully
- **Red dot**: Connection error (check configuration)

#### 2. Select CSV Files

**Required Files:**
- `organizations.csv` - Required
- `regions.csv` - Required
- `facilities.csv` - Required
- `providers.csv` - **Optional** (can be imported separately later)

**Option A: Load from Directory**
- Click "Load from Directory..."
- Select a folder containing the CSV files
- The application will automatically populate all available files
- If providers.csv is missing, the import will continue without it

**Option B: Select Individual Files**
- Click "Browse..." next to each field
- Select the corresponding CSV file for each entity type
- Providers file can be left empty if not needed

#### 3. Generate Templates (Optional)
If you don't have CSV files yet:
- Click "Generate Templates..."
- Choose a destination folder
- The application will create template CSV files with example data
- Edit the templates with your actual data
- Return to Step 2 to load your edited files

#### 4. Validate & Import
- Once all four files are selected, the "Validate & Import" button becomes enabled
- Click "Validate & Import"
- The application will:
  1. Parse all CSV files
  2. Validate data integrity
  3. Check referential integrity (regions reference orgs, facilities reference regions, etc.)
  4. Automatically import data if validation succeeds
- Watch the log panel for real-time progress

#### 5. Review Results
After import completes:
- View the results summary showing counts for each entity type
- Check the log panel for any warnings or errors
- Click "Import More Data" to load additional records

## CSV File Requirements

### Required Files
Three CSV files are required:
1. **organizations.csv** - Organization/health system data (Required)
2. **regions.csv** - Regional divisions within organizations (Required)
3. **facilities.csv** - Individual facilities/clinics (Required)

### Optional Files
4. **providers.csv** - Healthcare providers (Optional - can be imported separately)

### Data Validation
The application validates:
- **Required fields**: All mandatory columns must have values
- **Duplicate detection**: No duplicate codes or NPIs
- **Referential integrity**: Child records must reference valid parent records
- **Data format**: NPIs must be exactly 10 digits

See the [CLI README](../pro-data-loader/README.md) for detailed CSV format specifications.

## Configuration

The application automatically finds the database configuration from:
1. `C:\ProgramData\Professional SMART\config\.env`
2. `C:\Program Files\Professional SMART\config\.env`
3. `.env` (current directory)

The configuration file must contain:
```
DATABASE_URL=postgres://username:password@localhost:5432/professional_smart
```

## Import Behavior

- **Upsert**: Existing records are updated based on unique keys
- **Transaction**: All imports happen in a single transaction
- **Dependency Order**: Data is imported in the correct order automatically

## Error Handling

### Connection Errors
If you see a red connection indicator:
1. Verify PostgreSQL is running
2. Check the `.env` file exists and has correct DATABASE_URL
3. Test database connectivity using `pro-setup.exe`

### Validation Errors
The application will show:
- The exact row number where the error occurred
- A description of what's wrong
- Suggestions for fixing the issue

### Import Errors
If import fails after validation:
- All changes are rolled back (nothing is imported)
- Check the log panel for detailed error messages
- Fix the underlying issue and try again

## Log Panel

The bottom panel shows all operations:
- **Timestamp**: When the operation occurred
- **Level**: INFO, SUCCESS, WARNING, or ERROR
- **Message**: Detailed description

The log automatically scrolls to show the latest entries and keeps the last 100 entries.

## Tips

1. **Start with Templates**: Use "Generate Templates..." to see the required format
2. **Validate First**: The application validates before importing, so you can fix errors without affecting the database
3. **Check the Log**: The log panel provides detailed information about what's happening
4. **Incremental Loads**: You can run the import multiple times - existing records will be updated, new records will be added

## Troubleshooting

### Application Won't Start
- Check if all required DLLs are present
- Try running from command line to see error messages

### Files Won't Load
- Verify CSV files are not open in Excel or another program
- Check file permissions

### Validation Fails
- Review error messages for specific row numbers
- Compare your CSV format with generated templates
- Check for special characters or encoding issues

### Import Hangs
- Check database connection
- Verify PostgreSQL is not overloaded
- Check firewall settings if database is remote

## Technical Details

- **Framework**: egui (immediate-mode GUI)
- **Runtime**: Tokio (async Rust)
- **Database**: PostgreSQL via sqlx
- **File Dialogs**: Native Windows file picker (rfd)
- **Executable Size**: ~6MB (includes all dependencies)

## Comparison: GUI vs CLI

| Feature | GUI | CLI |
|---------|-----|-----|
| Ease of Use | Visual, intuitive | Command-line arguments |
| File Selection | Browse dialogs | Path specification |
| Progress Display | Real-time UI | Console output |
| Validation Feedback | Formatted panels | Text output |
| Template Generation | Button click | Subcommand |
| Confirmation | Automatic | Manual (yes/no) |
| Best For | Interactive use | Scripts, automation |

Both versions share the same core logic and validation rules.

## Support

For issues or questions:
1. Check the log panel for error details
2. Review the [main README](../../README.md) for database setup
3. Consult [DATABASE_SETUP.md](../../docs/DATABASE_SETUP.md) for PostgreSQL configuration
4. Report bugs to the Professional SMART team
