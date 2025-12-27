# CLAUDE.md

### Claude Rules
Rule 1: NEVER disable or remove a feature to fix a bug or error.
Rule 2: NEVER fix an error or bug by hiding it.
Rule 3: NO silent fallbacks or silent failures, all problems should be loud and proud.
Rule 4: Always check online documentation of every package used and do everything the officially recommended way.
Rule 5: Clean up your mess. Remove any temporary and/or outdated files or scripts that were only meant to be used once and no longer serve a purpose.
Rule 6: NEVER use docker, this is a Windows only application and doesn't use containers.
Rule 7: NEVER use character emoji's in any of the code or documentation.
Rule 8: Create a .md file with what your plan is to resolve issues or to develop new functionaility and put it in a checklist.
Rule 9:  No shortcuts - fully resolve issues by solving it the right way and not creating cascading failures elsewhere.
Rule 10: Rebuild the installer after every change.
Rule 11: Version the build after every change and determine if it's a minor or major version change.
Rule 12: Absolutely no manual fixes.
Rule 13: Keep track of changes in an .md file (all changes go into this one file to keep track)
Rule 14: Read CHANGELOG.md of previous changes.
Rule 15: Any new sql migrations need to be part of the 000 baseline

## Installer Build Process

### Build Command
```powershell
cd c:\Users\jonmc\dev\pro\installer
.\build-msi.ps1 -Version "X.Y.Z.W"
```

### Build Options
- `-Version "X.Y.Z.W"` - Specify version (required for version changes)
- `-NoBuild` - Skip Rust compilation (use existing binaries)

### Version Numbering (X.Y.Z.W)
- **X** (Major): Breaking changes, major rewrites
- **Y** (Minor): New features, new migrations
- **Z** (Patch): Bug fixes, performance improvements
- **W** (Build): Incremental builds

### Adding New Migrations
When adding a new migration (e.g., `068_new_feature.sql`):

1. **Create the migration file**: `migrations/068_new_feature.sql`

2. **Add to embedded migrations**: Edit `crates/pro-upgrade-manager/src/embedded_migrations.rs`:
   ```rust
   EmbeddedMigration {
       version: "068",
       name: "new_feature",
       sql: include_str!("../../../migrations/068_new_feature.sql"),
   },
   ```

3. **Update the baseline**: Append the migration SQL to `migrations/000_baseline_v2.12.sql`:
   ```sql
   -- ============================================================================
   -- Source: 068_new_feature.sql
   -- ============================================================================

   -- [migration content here]
   ```

4. **Update baseline header**: Change description line to include new migration range:
   ```sql
   -- Description: Complete schema baseline generated from migrations 001-068
   ```

5. **Update CHANGELOG.md**: Add entry for the new version

6. **Rebuild installer**: `.\build-msi.ps1 -Version "X.Y.Z.W"`

### Output
- MSI location: `c:\Users\jonmc\dev\pro\installer\ProfessionalSMART.msi`
- Version file: `c:\Users\jonmc\dev\pro\installer\version.txt`

## Collaboration Guidelines
- **Challenge and question**: Don't immediately agree or proceed with requests that seem suboptimal, unclear, or potentially problematic
- **Push back constructively**: If a proposed approach has issues, suggest better alternatives with clear reasoning
- **Think critically**: Consider edge cases, performance implications, maintainability, and best practices before implementing
- **Seek clarification**: Ask follow-up questions when requirements are ambiguous or could be interpreted multiple ways
- **Propose improvements**: Suggest better patterns, more robust solutions, or cleaner implementations when appropriate
- **Be a thoughtful collaborator**: Act as a good teammate who helps improve the overall quality and direction of the project