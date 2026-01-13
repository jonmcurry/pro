# Rule Converter Tool Plan

## Status: COMPLETE

## Overview
Create a GUI tool that connects to MS SQL Server, queries legacy filter rules, and converts them to COMPOSITE template SQL format.

## Phase 1: CLI Tool (COMPLETE)
- [x] Create `crates/pro-rule-converter/Cargo.toml`
- [x] Create `crates/pro-rule-converter/src/main.rs`
- [x] Parse legacy filter definitions
- [x] Convert to COMPOSITE template JSON
- [x] Output SQL INSERT statements

## Phase 2: GUI with MS SQL Connection (COMPLETE)

### Implementation
- [x] Create `crates/pro-rule-converter-gui/Cargo.toml`
- [x] Create NWG-based GUI application
- [x] Add MS SQL Server connectivity (tiberius crate)
- [x] Create config file for SQL query
- [x] Display rules in ListView with multi-select
- [x] Export selected rules to SQL file
- [x] Add to workspace Cargo.toml
- [x] Update CHANGELOG.md
- [x] Rebuild installer v2.12.73.0

### Files Created
- `crates/pro-rule-converter-gui/Cargo.toml`
- `crates/pro-rule-converter-gui/src/main.rs` - NWG GUI application
- `crates/pro-rule-converter-gui/src/converter.rs` - Rule parsing and SQL generation
- `crates/pro-rule-converter-gui/src/mssql.rs` - MS SQL Server client using tiberius
- `crates/pro-rule-converter-gui/rule-converter-config.toml` - Configuration file
- `crates/pro-rule-converter-gui/build.rs` - Windows resource build script
- `crates/pro-rule-converter-gui/windows-manifest.rc` - Windows manifest resource
- `crates/pro-rule-converter-gui/windows-manifest.xml` - DPI awareness manifest

### Config File Format
`rule-converter-config.toml`:
```toml
[database]
server = "localhost"
port = 1433
database = "FilterDB"
auth_type = "windows"  # or "sql"
username = ""
password = ""

[query]
sql = """
select a.FilterNumber, a.FilterName, FilterDescription,
       [dbo].[fn_DecryptFilterDefinition1](FilterDefinition) as 'definition'
from tblopfilter_filter a
join tblopfilter_filtergroup b on a.filtergroupid = b.filtergroupid
join tblopfilter_filterdefs c on a.filternumber = c.filternumber
where icdver = '10'
  and (a.filternumber not like 'RP%' and a.filternumber not like 'HOSP%')
  and c.enddate is null
"""

[output]
output_directory = "."
flag_category = "QM"
```

## Parser Mappings

| Legacy Parser | COMPOSITE Condition |
|---------------|---------------------|
| `Parser.In(DX, "codes")` | `dx_in` or `dx_pattern` |
| `Parser.In(CPT, "codes")` | `cpt_in` (expands ranges) |
| `Parser.In(DISCH_DATE, "range")` | `date_gte` |
| `Parser.In(POS, "codes")` | `pos_in` |
| `&&` in FDEF | `"operator": "AND"` |
| `||` in FDEF | `"operator": "OR"` |

## Version
2.12.73.0
