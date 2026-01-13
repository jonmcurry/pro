//! Rule Converter Tool
//!
//! Converts legacy filter rules to COMPOSITE template SQL format.
//!
//! Input format (tab-separated):
//!   RULE_CODE\tRULE_NAME\t\tDESCRIPTION\tFILTER_DEF
//!
//! Usage:
//!   pro-rule-converter input.txt > output.sql
//!   pro-rule-converter --inline "AHRQOP001A\tRule Name\t\tDescription\tFilterDef"

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use regex::Regex;
use serde::Serialize;
use std::fs;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "pro-rule-converter")]
#[command(about = "Converts legacy filter rules to COMPOSITE template SQL format")]
struct Args {
    /// Input file containing tab-separated rules (one per line)
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Output SQL file (defaults to stdout)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Single inline rule to convert (tab-separated)
    #[arg(long)]
    inline: Option<String>,

    /// Flag issue code to use (default: auto-generate from rule code)
    #[arg(long)]
    flag_issue: Option<String>,

    /// Flag category code (default: QM for quality measures)
    #[arg(long, default_value = "QM")]
    category: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
enum Condition {
    #[serde(rename = "cpt_in")]
    CptIn { codes: Vec<String> },

    #[serde(rename = "cpt_pattern")]
    CptPattern { pattern: String },

    #[serde(rename = "dx_in")]
    DxIn { codes: Vec<String> },

    #[serde(rename = "dx_pattern")]
    DxPattern { pattern: String },

    #[serde(rename = "dx_pattern_exclude")]
    DxPatternExclude { include: String, exclude: String },

    #[serde(rename = "date_gte")]
    DateGte { min_date: String },

    #[serde(rename = "date_lte")]
    DateLte { max_date: String },

    #[serde(rename = "pos_in")]
    PosIn { codes: Vec<String> },
}

#[derive(Debug, Clone, Serialize)]
struct CompositeParams {
    operator: String,
    conditions: Vec<Condition>,
}

#[derive(Debug, Clone)]
struct ParsedRule {
    rule_code: String,
    rule_name: String,
    description: String,
    conditions: Vec<Condition>,
    operator: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let rules = if let Some(inline) = &args.inline {
        vec![parse_rule_line(inline)?]
    } else if let Some(input_path) = &args.input {
        let content = fs::read_to_string(input_path)
            .with_context(|| format!("Failed to read input file: {:?}", input_path))?;
        parse_rules(&content)?
    } else {
        // Read from stdin
        let stdin = io::stdin();
        let lines: Vec<String> = stdin.lock().lines().collect::<Result<_, _>>()?;
        let content = lines.join("\n");
        parse_rules(&content)?
    };

    let sql = generate_sql(&rules, &args.category, args.flag_issue.as_deref())?;

    if let Some(output_path) = &args.output {
        fs::write(output_path, &sql)
            .with_context(|| format!("Failed to write output file: {:?}", output_path))?;
        eprintln!("Generated {} rule(s) to {:?}", rules.len(), output_path);
    } else {
        io::stdout().write_all(sql.as_bytes())?;
    }

    Ok(())
}

fn parse_rules(content: &str) -> Result<Vec<ParsedRule>> {
    let mut rules = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("--") {
            continue;
        }

        match parse_rule_line(line) {
            Ok(rule) => rules.push(rule),
            Err(e) => {
                eprintln!("Warning: Failed to parse line {}: {}", line_num + 1, e);
            }
        }
    }

    Ok(rules)
}

fn parse_rule_line(line: &str) -> Result<ParsedRule> {
    // Split by tab
    let parts: Vec<&str> = line.split('\t').collect();

    if parts.len() < 5 {
        return Err(anyhow!(
            "Expected 5 tab-separated fields, got {}: {:?}",
            parts.len(),
            parts
        ));
    }

    let rule_code = parts[0].trim().to_string();
    let rule_name = parts[1].trim().to_string();
    // parts[2] is empty
    let description = parts[3].trim().to_string();
    let filter_def = parts[4].trim();

    let (conditions, operator) = parse_filter_def(filter_def)?;

    Ok(ParsedRule {
        rule_code,
        rule_name,
        description,
        conditions,
        operator,
    })
}

fn parse_filter_def(filter_def: &str) -> Result<(Vec<Condition>, String)> {
    let mut conditions = Vec::new();
    let mut operator = "AND".to_string();

    // Remove trailing % if present
    let filter_def = filter_def.trim_end_matches('%').trim();

    // Parse each semicolon-separated part
    let parts: Vec<&str> = filter_def.split(';').collect();

    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        // Check for FDEF (defines the logical combination)
        if part.starts_with("FDEF=") {
            let fdef = &part[5..];
            if fdef.contains("||") {
                operator = "OR".to_string();
            } else {
                operator = "AND".to_string();
            }
            continue;
        }

        // Skip ID field
        if part.starts_with("ID=") {
            continue;
        }

        // Parse Parser.In expressions
        if let Some(condition) = parse_parser_in(part)? {
            conditions.push(condition);
        }
    }

    Ok((conditions, operator))
}

fn parse_parser_in(part: &str) -> Result<Option<Condition>> {
    // Match: FIELD=Parser.In(FIELD_NAME, "values")
    let re = Regex::new(r#"(\w+)=Parser\.In\((\w+),\s*"([^"]+)"\)"#)?;

    if let Some(caps) = re.captures(part) {
        let field_type = caps.get(2).unwrap().as_str().to_uppercase();
        let values = caps.get(3).unwrap().as_str();

        match field_type.as_str() {
            "DX" => {
                let codes = parse_code_list(values);
                // Check if we should use pattern matching
                if codes.iter().any(|c| c.contains('*') || c.len() < 5) {
                    // Use pattern - convert codes to regex
                    let pattern = codes_to_pattern(&codes);
                    Ok(Some(Condition::DxPattern { pattern }))
                } else {
                    Ok(Some(Condition::DxIn { codes }))
                }
            }
            "CPT" => {
                let codes = parse_code_list(values);
                if codes.iter().any(|c| c.contains('-')) {
                    // Expand ranges and use cpt_in
                    let expanded = expand_code_ranges(&codes);
                    Ok(Some(Condition::CptIn { codes: expanded }))
                } else {
                    Ok(Some(Condition::CptIn { codes }))
                }
            }
            "DISCH_DATE" | "DOS" | "SERVICE_DATE" => {
                // Parse date range like "20120701-99991231"
                if let Some((min_date, _max_date)) = parse_date_range(values) {
                    Ok(Some(Condition::DateGte { min_date }))
                } else {
                    Ok(None)
                }
            }
            "POS" => {
                let codes = parse_code_list(values);
                Ok(Some(Condition::PosIn { codes }))
            }
            _ => {
                eprintln!("Warning: Unknown field type: {}", field_type);
                Ok(None)
            }
        }
    } else {
        Ok(None)
    }
}

fn parse_code_list(values: &str) -> Vec<String> {
    values
        .split(',')
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty())
        .collect()
}

fn expand_code_ranges(codes: &[String]) -> Vec<String> {
    let mut result = Vec::new();

    for code in codes {
        if code.contains('-') {
            let parts: Vec<&str> = code.split('-').collect();
            if parts.len() == 2 {
                if let (Ok(start), Ok(end)) = (
                    parts[0].trim().parse::<u32>(),
                    parts[1].trim().parse::<u32>(),
                ) {
                    for i in start..=end {
                        result.push(format!("{:05}", i));
                    }
                    continue;
                }
            }
        }
        result.push(code.clone());
    }

    result
}

fn codes_to_pattern(codes: &[String]) -> String {
    // Convert list of codes to regex pattern
    // e.g., ["F1110", "F1120"] -> "^(F1110|F1120)"
    // e.g., ["F11"] -> "^F11" (prefix match)

    if codes.len() == 1 {
        let code = &codes[0];
        if code.len() <= 3 {
            // Short code - treat as prefix
            format!("^{}", code)
        } else {
            format!("^{}$", code)
        }
    } else {
        // Check if all codes share a common prefix
        let first = &codes[0];
        let common_prefix_len = codes
            .iter()
            .skip(1)
            .fold(first.len(), |len, code| {
                first
                    .chars()
                    .zip(code.chars())
                    .take(len)
                    .take_while(|(a, b)| a == b)
                    .count()
            });

        if common_prefix_len >= 2 && codes.iter().all(|c| c.len() <= common_prefix_len + 2) {
            // All codes share a significant prefix - use prefix pattern
            format!("^{}", &first[..common_prefix_len])
        } else {
            // Use alternation
            let alternatives: Vec<String> = codes.iter().map(|c| c.to_string()).collect();
            format!("^({})", alternatives.join("|"))
        }
    }
}

fn parse_date_range(values: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = values.split('-').collect();
    if parts.len() == 2 {
        let min_date = format_date(parts[0])?;
        let max_date = format_date(parts[1])?;
        Some((min_date, max_date))
    } else {
        None
    }
}

fn format_date(date_str: &str) -> Option<String> {
    // Convert YYYYMMDD to YYYY-MM-DD
    if date_str.len() == 8 {
        let year = &date_str[0..4];
        let month = &date_str[4..6];
        let day = &date_str[6..8];
        Some(format!("{}-{}-{}", year, month, day))
    } else {
        None
    }
}

fn generate_sql(rules: &[ParsedRule], category: &str, flag_issue_override: Option<&str>) -> Result<String> {
    let mut sql = String::new();

    sql.push_str("-- Generated by pro-rule-converter\n");
    sql.push_str("-- COMPOSITE template rules for the Professional SMART rules engine\n");
    sql.push_str("-- \n\n");

    for rule in rules {
        let flag_issue_code = flag_issue_override
            .map(|s| s.to_string())
            .unwrap_or_else(|| generate_flag_issue_code(&rule.rule_code));

        let params = CompositeParams {
            operator: rule.operator.clone(),
            conditions: rule.conditions.clone(),
        };

        let params_json = serde_json::to_string(&params)?;

        // Escape single quotes in description
        let escaped_desc = rule.description.replace('\'', "''");
        let escaped_name = rule.rule_name.replace('\'', "''");

        sql.push_str(&format!(
            r#"-- Rule: {} - {}
-- Ensure flag category exists
INSERT INTO claims.flag_category (category_code, category_name, category_description)
SELECT '{category}', 'Quality Measures', 'AHRQ and other quality measure indicators'
WHERE NOT EXISTS (SELECT 1 FROM claims.flag_category WHERE category_code = '{category}');

-- Ensure flag issue exists
INSERT INTO claims.flag_issue (category_id, issue_code, issue_description, severity)
SELECT
    (SELECT category_id FROM claims.flag_category WHERE category_code = '{category}'),
    '{flag_issue_code}',
    '{escaped_name}',
    'MEDIUM'
WHERE NOT EXISTS (SELECT 1 FROM claims.flag_issue WHERE issue_code = '{flag_issue_code}');

-- Rule definition
INSERT INTO claims.rule_definition (
    rule_code,
    rule_name,
    rule_description,
    template_id,
    rule_parameters_encrypted,
    flag_issue_id,
    execution_order,
    execution_level,
    default_severity,
    timeout_ms,
    is_active
)
SELECT
    '{rule_code}',
    '{escaped_name}',
    '{escaped_desc}',
    (SELECT template_id FROM claims.rule_template WHERE template_code = 'COMPOSITE'),
    pgp_sym_encrypt(
        '{params_json}',
        current_setting('app.rule_encryption_key', true)
    ),
    (SELECT issue_id FROM claims.flag_issue WHERE issue_code = '{flag_issue_code}'),
    100,
    'SERVICE_LINE',
    'MEDIUM',
    5000,
    true
WHERE NOT EXISTS (SELECT 1 FROM claims.rule_definition WHERE rule_code = '{rule_code}');

"#,
            rule.rule_code,
            rule.rule_name,
            category = category,
            flag_issue_code = flag_issue_code,
            rule_code = rule.rule_code,
            escaped_name = escaped_name,
            escaped_desc = escaped_desc,
            params_json = params_json,
        ));
    }

    Ok(sql)
}

fn generate_flag_issue_code(rule_code: &str) -> String {
    // Generate a flag issue code from rule code
    // e.g., AHRQOP001A -> QM_AHRQOP001A
    format!("QM_{}", rule_code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_code_list() {
        let codes = parse_code_list("F1110,F1120,F1130");
        assert_eq!(codes, vec!["F1110", "F1120", "F1130"]);
    }

    #[test]
    fn test_expand_code_ranges() {
        let codes = vec!["99281-99285".to_string(), "99291".to_string()];
        let expanded = expand_code_ranges(&codes);
        assert_eq!(
            expanded,
            vec!["99281", "99282", "99283", "99284", "99285", "99291"]
        );
    }

    #[test]
    fn test_format_date() {
        assert_eq!(format_date("20120701"), Some("2012-07-01".to_string()));
    }

    #[test]
    fn test_codes_to_pattern() {
        let codes = vec!["F1110".to_string(), "F1120".to_string()];
        let pattern = codes_to_pattern(&codes);
        assert_eq!(pattern, "^F11"); // Common prefix
    }

    #[test]
    fn test_parse_rule_line() {
        let line = "AHRQOP001A\tOpiod Related Hospital Visits\t\tDescription here\tID=AHRQOP001A;DX=Parser.In(DX,\"F1110,F1120\");CPT=Parser.In(CPT,\"99281-99285\");FDEF=DX && CPT;%";
        let rule = parse_rule_line(line).unwrap();
        assert_eq!(rule.rule_code, "AHRQOP001A");
        assert_eq!(rule.operator, "AND");
        assert_eq!(rule.conditions.len(), 2);
    }
}
