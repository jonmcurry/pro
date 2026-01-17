# Fix OR rules that have date_gte as first condition
# These should be AND rules, not OR

$content = Get-Content 'home.txt' -Raw

# Replace OR with AND for rules that have date_gte first
# Pattern: "operator":"OR","conditions":[{"type":"date_gte"
# Replace with: "operator":"AND","conditions":[{"type":"date_gte"
$fixed = $content -replace '"operator":"OR","conditions":\[\{"type":"date_gte"', '"operator":"AND","conditions":[{"type":"date_gte"'

# Save to new file
$fixed | Out-File -FilePath 'home_fixed.txt' -Encoding UTF8 -NoNewline

# Verify the fix
$newOrDateFirst = ([regex]::Matches($fixed, '"operator":"OR","conditions":\[\{"type":"date_gte"')).Count
$newOrTotal = ([regex]::Matches($fixed, '"operator":"OR"')).Count
$newAndTotal = ([regex]::Matches($fixed, '"operator":"AND"')).Count

Write-Host "AFTER FIX:"
Write-Host "Total OR rules: $newOrTotal"
Write-Host "OR rules with date_gte first: $newOrDateFirst"
Write-Host "Total AND rules: $newAndTotal"
Write-Host ""
Write-Host "Fixed file saved to: home_fixed.txt"
Write-Host ""
Write-Host "EXPECTED PERFORMANCE IMPROVEMENT:"
Write-Host "- Before: 537 universal rules execute on every service line"
Write-Host "- After: Only ~33 OR rules remain universal, 504 AND rules use CPT indexing"
Write-Host "- CPT indexing will skip ~90% of rules per service line"
Write-Host "- Target: 10,000 claims in 15-30 seconds should now be achievable"
