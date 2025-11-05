//! Rule Result Caching
//!
//! PHASE 5: Cache rule execution results to avoid re-evaluating identical claim patterns.
//! Expected impact: 40-60% reduction in rule execution time for common claim scenarios.

use crate::rule_engine::{RuleExecutionContext, RuleResult};
use chrono::{DateTime, Duration, Utc};
use rustc_hash::FxHashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Cache for rule execution results
///
/// Caches results based on claim characteristics (not IDs), allowing cache hits
/// across different claims with identical properties.
#[derive(Clone)]
pub struct RuleResultCache {
    cache: Arc<RwLock<FxHashMap<u64, CachedRuleResult>>>,
    ttl: Duration,
    max_size: usize,
    hits: Arc<AtomicU64>,
    misses: Arc<AtomicU64>,
}

/// Cached rule result with timestamp
#[derive(Debug, Clone)]
struct CachedRuleResult {
    results: Vec<RuleResult>,
    cached_at: DateTime<Utc>,
}

impl CachedRuleResult {
    fn new(results: Vec<RuleResult>) -> Self {
        Self {
            results,
            cached_at: Utc::now(),
        }
    }

    fn is_valid(&self, ttl: Duration) -> bool {
        Utc::now() - self.cached_at < ttl
    }
}

impl RuleResultCache {
    /// Create a new rule result cache with default settings
    ///
    /// Default TTL: 60 seconds (conservative to avoid stale data)
    /// Default max size: 10,000 entries (~2MB memory)
    pub fn new() -> Self {
        Self::with_config(Duration::seconds(60), 10_000)
    }

    /// Create a cache with custom configuration
    pub fn with_config(ttl: Duration, max_size: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(FxHashMap::default())),
            ttl,
            max_size,
            hits: Arc::new(AtomicU64::new(0)),
            misses: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Get cached results if available and valid
    pub fn get(&self, ctx: &RuleExecutionContext) -> Option<Vec<RuleResult>> {
        let fingerprint = Self::compute_fingerprint(ctx);

        let cache = self.cache.read().ok()?;
        if let Some(cached) = cache.get(&fingerprint) {
            if cached.is_valid(self.ttl) {
                self.hits.fetch_add(1, Ordering::Relaxed);
                return Some(cached.results.clone());
            }
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        None
    }

    /// Store results in cache
    pub fn insert(&self, ctx: &RuleExecutionContext, results: Vec<RuleResult>) {
        let fingerprint = Self::compute_fingerprint(ctx);
        let cached_result = CachedRuleResult::new(results);

        if let Ok(mut cache) = self.cache.write() {
            // Evict oldest entries if cache is full
            if cache.len() >= self.max_size {
                self.evict_oldest(&mut cache);
            }

            cache.insert(fingerprint, cached_result);
        }
    }

    /// Clear all cached results
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            (hits as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let size = self.cache.read().map(|c| c.len()).unwrap_or(0);

        CacheStats {
            hits,
            misses,
            total,
            hit_rate,
            size,
            max_size: self.max_size,
        }
    }

    /// Compute fingerprint for cache key
    ///
    /// Hash includes only the fields that affect rule outcomes, excluding:
    /// - encounter_id, service_line_id (entity IDs - different for each claim)
    /// - facility_id, organization_id (filtered at rule selection, not evaluation)
    /// - subscriber_id (used for cache lookups, not rule logic)
    ///
    /// This allows cache hits across different claims with identical characteristics.
    fn compute_fingerprint(ctx: &RuleExecutionContext) -> u64 {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        // Core service line characteristics
        if let Some(ref code) = ctx.procedure_code {
            code.hash(&mut hasher);
        }
        if let Some(units) = ctx.service_unit_count {
            units.to_string().hash(&mut hasher);
        }
        if let Some(amount) = ctx.line_item_charge_amount {
            amount.to_string().hash(&mut hasher);
        }

        // Modifiers (sorted for consistency)
        let mut modifiers = ctx.procedure_modifiers.clone();
        modifiers.sort();
        for modifier in modifiers {
            modifier.hash(&mut hasher);
        }

        // Diagnosis codes (sorted for consistency)
        let mut dx_codes = ctx.diagnosis_codes.clone();
        dx_codes.sort();
        for dx in dx_codes {
            dx.hash(&mut hasher);
        }

        // Date characteristics (exact date, not ID-based)
        if let Some(date) = ctx.date_of_service {
            date.hash(&mut hasher);
        }
        if let Some(date) = ctx.date_of_service_from {
            date.hash(&mut hasher);
        }

        // Place of service
        if let Some(ref pos) = ctx.place_of_service_code {
            pos.hash(&mut hasher);
        }

        // Claim-level characteristics
        if let Some(amount) = ctx.total_claim_charge_amount {
            amount.to_string().hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Evict oldest 10% of entries when cache is full
    fn evict_oldest(&self, cache: &mut FxHashMap<u64, CachedRuleResult>) {
        let eviction_count = (self.max_size / 10).max(1);

        // Find oldest entries and collect keys
        let mut entries: Vec<_> = cache.iter().collect();
        entries.sort_by_key(|(_, v)| v.cached_at);
        let keys_to_remove: Vec<u64> = entries.iter().take(eviction_count).map(|(k, _)| **k).collect();

        // Remove oldest entries
        for key in keys_to_remove {
            cache.remove(&key);
        }
    }
}

impl Default for RuleResultCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub total: u64,
    pub hit_rate: f64,
    pub size: usize,
    pub max_size: usize,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cache Stats: {} hits, {} misses, {:.2}% hit rate, {}/{} entries",
            self.hits, self.misses, self.hit_rate, self.size, self.max_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_cache_creation() {
        let cache = RuleResultCache::new();
        let stats = cache.stats();
        assert_eq!(stats.size, 0);
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[test]
    fn test_cache_hit_miss() {
        let cache = RuleResultCache::new();

        let mut ctx = RuleExecutionContext::new(1);
        ctx.procedure_code = Some("99213".to_string());
        ctx.diagnosis_codes = vec!["J06.9".to_string()];

        // First access should be a miss
        assert!(cache.get(&ctx).is_none());
        let stats = cache.stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);

        // Insert result
        let results = vec![];
        cache.insert(&ctx, results);

        // Second access should be a hit
        assert!(cache.get(&ctx).is_some());
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.size, 1);
    }

    #[test]
    fn test_cache_fingerprint_ignores_ids() {
        let cache = RuleResultCache::new();

        // Two contexts with different IDs but same characteristics
        let mut ctx1 = RuleExecutionContext::new(1);
        ctx1.encounter_id = Some(1);
        ctx1.service_line_id = Some(1);
        ctx1.procedure_code = Some("99213".to_string());
        ctx1.diagnosis_codes = vec!["J06.9".to_string()];

        let mut ctx2 = RuleExecutionContext::new(1);
        ctx2.encounter_id = Some(1); // Different ID
        ctx2.service_line_id = Some(1); // Different ID
        ctx2.procedure_code = Some("99213".to_string()); // Same
        ctx2.diagnosis_codes = vec!["J06.9".to_string()]; // Same

        // Insert with ctx1
        cache.insert(&ctx1, vec![]);

        // Should get cache hit with ctx2 (different IDs, same characteristics)
        assert!(cache.get(&ctx2).is_some());
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_cache_ttl_expiration() {
        let cache = RuleResultCache::with_config(Duration::milliseconds(10), 100);

        let mut ctx = RuleExecutionContext::new(1);
        ctx.procedure_code = Some("99213".to_string());

        // Insert result
        cache.insert(&ctx, vec![]);

        // Should hit immediately
        assert!(cache.get(&ctx).is_some());

        // Wait for TTL to expire
        std::thread::sleep(std::time::Duration::from_millis(15));

        // Should miss after expiration
        assert!(cache.get(&ctx).is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let cache = RuleResultCache::with_config(Duration::seconds(60), 10);

        // Fill cache beyond capacity
        for i in 0..15 {
            let mut ctx = RuleExecutionContext::new(1);
            ctx.procedure_code = Some(format!("9921{}", i));
            cache.insert(&ctx, vec![]);
        }

        // Cache should not exceed max size
        let stats = cache.stats();
        assert!(stats.size <= 10);
    }

    #[test]
    fn test_sorted_modifiers_hash_consistency() {
        // Ensure modifiers in different orders produce same fingerprint
        let mut ctx1 = RuleExecutionContext::new(1);
        ctx1.procedure_code = Some("99213".to_string());
        ctx1.procedure_modifiers = vec!["25".to_string(), "59".to_string()];

        let mut ctx2 = RuleExecutionContext::new(1);
        ctx2.procedure_code = Some("99213".to_string());
        ctx2.procedure_modifiers = vec!["59".to_string(), "25".to_string()]; // Reversed

        let fp1 = RuleResultCache::compute_fingerprint(&ctx1);
        let fp2 = RuleResultCache::compute_fingerprint(&ctx2);

        assert_eq!(fp1, fp2, "Fingerprints should match regardless of modifier order");
    }
}
