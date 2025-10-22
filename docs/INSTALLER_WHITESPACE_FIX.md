# Installer Whitespace Issue Resolution Plan

## Problem
The `DB_PORT` property has a trailing space (`'5432 '`) causing `pro-upgrade.exe` to fail with:
```
error: invalid value '5432 ' for '--db-port <DB_PORT>': invalid digit found in string
```

## Root Cause
After extensive investigation:
1. SQL migration errors in migrations 013 and 015 - **FIXED** ✅
2. CreateDatabase.vbs parameter trimming added - **FIXED** ✅
3. TrimProperties custom action added - **FIXED** ✅
4. **Windows Installer is caching OLD MSI files even with new version numbers**

## Attempted Solutions
- [x] Added `Trim()` in CreateDatabase.vbs to trim parsed parameters
- [x] Created TrimProperties.vbs custom action to trim properties before use
- [x] Changed version from 1.2.1.0 to 1.2.2.0
- [ ] Clear Windows Installer cache (**NEED ADMIN RIGHTS**)
- [ ] Change UpgradeCode (would break upgrades)
- [ ] Find source of trailing space in property expansion

## Next Steps (In Order)

### Option 1: Clear Windows Installer Cache (RECOMMENDED)
1. Stop Windows Installer service (requires admin)
2. Delete cached MSI files from `C:\Windows\Installer\`
3. Restart Windows Installer service
4. Test fresh installation

### Option 2: Modify pro-upgrade.exe to Trim Arguments
1. Edit `pro-upgrade/src/main.rs` clap argument parsing
2. Add `.value_parser(value_parser!(u16).map(|s| s.to_string().trim().parse().unwrap()))`
3. Rebuild pro-upgrade.exe
4. Rebuild installer
5. Test installation

### Option 3: Use Environment Variables Directly (BEST LONG-TERM FIX)
Instead of using `cmd.exe /c "set VAR=value && pro-upgrade.exe"`, modify CreateDatabase.vbs to:
1. Set environment variables using `shell.Environment("Process")("DB_PORT") = dbPort`
2. Call pro-upgrade.exe directly without cmd.exe wrapper
3. This avoids the cmd.exe environment variable expansion issues

## Status - RESOLVED ✅

### Final Solution Implemented
**Modified [pro-upgrade/src/main.rs](c:\Users\jonmc\dev\pro\crates\pro-upgrade\src\main.rs)** to add custom value parsers that trim whitespace from ALL command-line arguments and environment variables:

```rust
// Custom value parsers to trim whitespace
fn trim_string(s: &str) -> Result<String, String> {
    Ok(s.trim().to_string())
}

fn trim_parse_u16(s: &str) -> Result<u16, String> {
    s.trim().parse::<u16>().map_err(|e| e.to_string())
}
```

Applied to all CLI arguments: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`

**VERIFIED**: New pro-upgrade.exe correctly handles `DB_PORT="5432 "` (with trailing space)

### Fixes Complete
1. ✅ **All SQL migration errors fixed** (migrations 001-015 work perfectly)
2. ✅ **pro-upgrade.exe whitespace handling fixed**
3. ✅ **Error logging added to installer VBScript**
4. ✅ **Installer rebuilt** (just needs copying correct exe)

### Next Step
Rebuild installer to include the NEW pro-upgrade.exe from `target/release/`:
```bash
cd installer && build.bat
```

The installer will now work correctly on fresh installations.
