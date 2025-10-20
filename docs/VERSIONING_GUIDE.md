# Versioning Guide

## Version Number Format

Professional SMART uses **Semantic Versioning** with a build number:

```
MAJOR.MINOR.PATCH.BUILD
```

Example: `1.2.3.456`

- **MAJOR**: Breaking changes, major features, schema changes requiring migration
- **MINOR**: New features, non-breaking enhancements
- **PATCH**: Bug fixes, minor improvements
- **BUILD**: Automatic increment for each build

## When to Increment Each Number

### MAJOR Version (x.0.0.0)

Increment when you have:

- **Breaking Changes**: Changes that require users to modify their workflow
- **Major Database Changes**: Schema changes that can't be migrated automatically
- **API Breaking Changes**: Changes to REST endpoints that break compatibility
- **Major Feature Overhauls**: Complete redesign of core functionality
- **Migration from Legacy**: Significant upgrade requiring manual intervention

**Examples:**
- `1.x.x.x → 2.0.0.0` - Complete UI redesign
- `1.x.x.x → 2.0.0.0` - Change from PostgreSQL to different database
- `1.x.x.x → 2.0.0.0` - Remove support for 837p v4010 (breaking for some users)

**What to do:**
```cmd
cd installer
.\build-simple.bat -Major
```

**⚠️ IMPORTANT:** Major version changes should:
- Include migration guide in release notes
- Warn users about breaking changes
- Provide rollback documentation
- Test extensively before release

---

### MINOR Version (1.x.0.0)

Increment when you add:

- **New Features**: New functionality that doesn't break existing features
- **New Migrations**: Database schema additions (new tables/columns)
- **New API Endpoints**: Additional REST endpoints (existing ones unchanged)
- **Performance Improvements**: Significant performance enhancements
- **New Data Sources**: Support for new file formats
- **New Reports/Dashboards**: Additional analytics capabilities

**Examples:**
- `1.1.x.x → 1.2.0.0` - Add denial prediction ML model
- `1.2.x.x → 1.3.0.0` - Add support for institutional (837i) claims
- `1.3.x.x → 1.4.0.0` - Add real-time claims validation API
- `1.4.x.x → 1.5.0.0` - Add multi-tenant support

**What to do:**
```cmd
cd installer
.\build-simple.bat -Minor
```

**Migration Impact:**
- New database migrations will be applied automatically
- Existing functionality continues to work
- Users can upgrade without data loss
- Backward compatible with previous minor versions

---

### PATCH Version (1.1.x.0)

Increment when you fix:

- **Bug Fixes**: Corrections to existing functionality
- **Security Patches**: Security vulnerability fixes
- **Performance Tweaks**: Minor performance improvements
- **UI Refinements**: Small UI/UX improvements
- **Documentation Updates**: Corrections to docs or help text
- **Configuration Improvements**: Better defaults or validation

**Examples:**
- `1.1.0.x → 1.1.1.0` - Fix denial reason not saving correctly
- `1.1.1.x → 1.1.2.0` - Fix memory leak in CSV parser
- `1.1.2.x → 1.1.3.0` - Correct RVU calculation for modifier 26
- `1.1.3.x → 1.1.4.0` - Fix service crash on malformed 837p file

**What to do:**
```cmd
cd installer
.\build-simple.bat -Patch
```

**Migration Impact:**
- Usually no database changes
- Drop-in replacement for same minor version
- Critical patches should be deployed quickly

---

### BUILD Number (1.1.0.x)

Automatically incremented for:

- **Development Builds**: Each time you build the MSI
- **Test Releases**: Internal testing before official release
- **CI/CD Builds**: Continuous integration builds
- **Hotfix Iterations**: Multiple attempts at fixing same issue

**Examples:**
- `1.1.0.1` - First build
- `1.1.0.2` - Fixed build issue, rebuild
- `1.1.0.3` - Added forgotten file, rebuild
- `1.1.0.100` - 100th development build

**What to do:**
```cmd
cd installer
.\build-simple.bat
# Automatically increments build number
```

**Note:** Build number resets to 0 when MAJOR, MINOR, or PATCH is incremented.

---

## Practical Guidelines

### Starting a New Feature Branch

```cmd
# Create feature branch
git checkout -b feature/denial-prediction

# Work on feature...
# Build and test locally
cd installer
.\build-simple.bat  # Increments build: 1.1.0.1, 1.1.0.2, etc.

# When feature is complete and ready for release
.\build-simple.bat -Minor  # Bumps to 1.2.0.0
```

### Hotfix for Production Bug

```cmd
# Create hotfix branch
git checkout -b hotfix/rvu-calculation-fix

# Fix the bug
# Build and test
cd installer
.\build-simple.bat  # Test build: 1.1.3.1

# When verified
.\build-simple.bat -Patch  # Release as 1.1.4.0
```

### Quarterly Major Release

```cmd
# Plan major release with multiple features
# Feature 1: Add 837i support
.\build-simple.bat -Minor  # → 1.5.0.0

# Feature 2: Add multi-tenant
.\build-simple.bat -Minor  # → 1.6.0.0

# Ready for major release with breaking changes
.\build-simple.bat -Major  # → 2.0.0.0
```

### Quick Rebuild (Same Version)

If you need to rebuild without changing version (e.g., forgot a file):

```cmd
# Rebuild without version change
cd installer
.\build-simple.bat -NoBuild  # Uses existing binaries
# OR manually edit version.txt before building
```

---

## Version History Examples

Here's how the version numbers might progress:

```
1.0.0.0   - Initial release
1.0.0.1   - Development build
1.0.0.2   - Development build
1.0.1.0   - Bugfix: Service startup issue
1.0.2.0   - Bugfix: CSV parser crash
1.1.0.0   - Feature: Upgrade path support (this release)
1.1.0.1   - Development build
1.1.1.0   - Bugfix: Upgrade detection issue
1.2.0.0   - Feature: Add denial prediction ML
1.2.0.1   - Development build
1.2.1.0   - Bugfix: ML model loading error
1.3.0.0   - Feature: Add 837i institutional claims
1.3.0.1   - Development build
1.3.1.0   - Bugfix: 837i parsing edge case
2.0.0.0   - BREAKING: New PostgreSQL schema, requires manual migration
2.0.0.1   - Development build
2.0.1.0   - Bugfix: Migration script error
2.1.0.0   - Feature: Real-time API
```

---

## Pre-Release Versions (Optional)

For alpha/beta releases, consider using a suffix:

```
1.2.0-alpha.1   - Alpha testing
1.2.0-alpha.2   - Alpha fixes
1.2.0-beta.1    - Beta testing
1.2.0-beta.2    - Beta fixes
1.2.0-rc.1      - Release candidate
1.2.0.0         - Final release
```

This requires modifying the build script to support pre-release tags.

---

## Decision Tree

Use this flowchart to decide which version number to increment:

```
Does this change break backward compatibility?
├─ YES → MAJOR version (x.0.0.0)
└─ NO
   └─ Does this add new functionality?
      ├─ YES → MINOR version (1.x.0.0)
      └─ NO
         └─ Is this a bug fix or minor improvement?
            ├─ YES → PATCH version (1.1.x.0)
            └─ NO
               └─ Is this just a rebuild?
                  └─ YES → BUILD number (1.1.0.x)
```

---

## Version in Git

Tag releases in Git with version numbers:

```cmd
# After building final release
git tag -a v1.2.0 -m "Release version 1.2.0 - Denial Prediction ML"
git push origin v1.2.0
```

---

## Checking Current Version

```cmd
# From version file
type installer\version.txt

# From registry (after install)
reg query "HKLM\SOFTWARE\ProfessionalSMART" /v Version

# From database
cd "C:\Program Files\Professional SMART\bin"
pro-upgrade.exe check-version

# From MSI file properties
# Right-click ProfessionalSMART.msi → Properties → Details
```

---

## Best Practices

1. **Always increment something** - Never release the same version twice
2. **Document changes** - Update changelog with each version
3. **Test upgrades** - Test upgrade path from previous version
4. **Tag in Git** - Create git tag for each release
5. **Keep version.txt in sync** - Commit version.txt after each release
6. **Build numbers are throwaway** - Don't rely on specific build numbers
7. **Reset build to 0** - When incrementing major/minor/patch
8. **Use meaningful names** - Name installers: `ProfessionalSMART-1.2.0.msi`

---

## Release Checklist

Before releasing a new version:

- [ ] Determine version number (major/minor/patch)
- [ ] Update CHANGELOG.md with changes
- [ ] Run full test suite
- [ ] Test upgrade from previous version
- [ ] Update documentation if needed
- [ ] Build release MSI
- [ ] Test MSI installation
- [ ] Create git tag
- [ ] Push to repository
- [ ] Create release notes
- [ ] Notify users (if applicable)

---

## FAQ

**Q: Can I skip version numbers?**
A: Yes, but not recommended. Going from 1.1.0 → 1.3.0 is confusing. Use sequential numbering.

**Q: What if I make a mistake and need to rebuild?**
A: Increment the build number (1.2.0.1 → 1.2.0.2). Don't reuse version numbers.

**Q: Should I increment for documentation-only changes?**
A: No, unless you're rebuilding the MSI. Pure documentation updates don't need new versions.

**Q: How do I handle hotfixes on old versions?**
A: Create a branch from the old version tag, apply fix, increment patch. Example: Branch from v1.2.0, fix bug, release as 1.2.1.

**Q: What about pre-release versions?**
A: Use the build number for pre-releases, or add suffixes like 1.2.0-beta.1. The current script doesn't support this, but you can manually set the version.

**Q: Can I go backward in version numbers?**
A: Never! The installer will reject downgrades. Always go forward.
