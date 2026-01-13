//! Rule conversion logic - parses legacy filter definitions and generates COMPOSITE SQL

use anyhow::{anyhow, Result};
use regex::Regex;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum Condition {
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
pub struct CompositeParams {
    pub operator: String,
    pub conditions: Vec<Condition>,
}

#[derive(Debug, Clone)]
pub struct ParsedRule {
    pub rule_code: String,
    pub rule_name: String,
    pub description: String,
    pub conditions: Vec<Condition>,
    pub operator: String,
}

pub fn parse_filter_def(filter_def: &str) -> Result<(Vec<Condition>, String)> {
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
                if codes.iter().any(|c| c.contains('*') || c.len() < 5) {
                    let pattern = codes_to_pattern(&codes);
                    Ok(Some(Condition::DxPattern { pattern }))
                } else {
                    Ok(Some(Condition::DxIn { codes }))
                }
            }
            "CPT" => {
                let codes = parse_code_list(values);
                if codes.iter().any(|c| c.contains('-')) {
                    let expanded = expand_code_ranges(&codes);
                    Ok(Some(Condition::CptIn { codes: expanded }))
                } else {
                    Ok(Some(Condition::CptIn { codes }))
                }
            }
            "DISCH_DATE" | "DOS" | "SERVICE_DATE" => {
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
            _ => Ok(None),
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
    if codes.len() == 1 {
        let code = &codes[0];
        if code.len() <= 3 {
            format!("^{}", code)
        } else {
            format!("^{}$", code)
        }
    } else {
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
            format!("^{}", &first[..common_prefix_len])
        } else {
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
    if date_str.len() == 8 {
        let year = &date_str[0..4];
        let month = &date_str[4..6];
        let day = &date_str[6..8];
        Some(format!("{}-{}-{}", year, month, day))
    } else {
        None
    }
}

pub fn generate_sql_for_rule(
    rule_code: &str,
    rule_name: &str,
    description: &str,
    filter_def: &str,
    category: &str,
) -> Result<String> {
    let (conditions, operator) = parse_filter_def(filter_def)?;

    if conditions.is_empty() {
        return Err(anyhow!("No valid conditions found in filter definition"));
    }

    let params = CompositeParams {
        operator,
        conditions,
    };

    let params_json = serde_json::to_string(&params)?;
    let flag_issue_code = format!("QM_{}", rule_code);

    let escaped_name = rule_name.replace('\'', "''");
    let escaped_desc = description.replace('\'', "''");

    Ok(format!(
        r#"-- Rule: {rule_code} - {rule_name}
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
        rule_code = rule_code,
        rule_name = rule_name,
        category = category,
        flag_issue_code = flag_issue_code,
        escaped_name = escaped_name,
        escaped_desc = escaped_desc,
        params_json = params_json,
    ))
}
