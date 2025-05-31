# src/cache/cache_manager.py
"""
Multi-Level Caching System for EDI Processing
"""
import logging
import time
import threading
import pickle
import hashlib
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from collections import OrderedDict
import psutil


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_time: float
    access_count: int = 0
    last_access: float = 0
    size_bytes: int = 0


class LRUCache:
    """Thread-safe LRU cache with size limits."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.total_size_bytes = 0
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.access_count += 1
                entry.last_access = time.time()
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                return entry.value
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put value in cache."""
        with self._lock:
            # Estimate size
            try:
                size_bytes = len(pickle.dumps(value))
            except pickle.PicklingError as e:
                self.logger.warning(
                    f"Could not pickle value for key '{key}' to estimate size: {e}. Using default size 1024 bytes."
                )
                size_bytes = 1024  # Default estimate
            except Exception as e: # Catch other unexpected errors during pickling
                self.logger.error(
                    f"Unexpected error pickling value for key '{key}': {e}. Using default size 1024 bytes.", exc_info=True
                )
                size_bytes = 1024
            
            # Check if value is too large
            if size_bytes > self.max_memory_bytes:
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.total_size_bytes -= old_entry.size_bytes
                del self.cache[key]
            
            # Ensure we have space
            self._ensure_space(size_bytes)
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                created_time=time.time(),
                size_bytes=size_bytes,
                last_access=time.time()
            )
            
            self.cache[key] = entry
            self.total_size_bytes += size_bytes
            
            return True
    
    def _ensure_space(self, needed_bytes: int):
        """Ensure sufficient space in cache."""
        while (len(self.cache) >= self.max_size or 
               self.total_size_bytes + needed_bytes > self.max_memory_bytes):
            if not self.cache:
                break
                
            # Remove least recently used item
            oldest_key, oldest_entry = self.cache.popitem(last=False)
            self.total_size_bytes -= oldest_entry.size_bytes

    def delete(self, key: str) -> bool:
        """Delete an item from the cache."""
        with self._lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.total_size_bytes -= entry.size_bytes
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.total_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.total_size_bytes / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024)
            }


class DistributedCache:
    """Redis-based distributed cache (optional)."""
    
    def __init__(self, redis_url: str = None):
        self.redis_client = None
        self.logger = logging.getLogger(__name__)
        
        if redis_url:
            try:
                import redis
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                self.logger.info("Redis distributed cache initialized")
            except Exception as e:
                self.logger.warning(f"Redis not available: {str(e)}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from distributed cache."""
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return pickle.loads(cached_data)
        except Exception as e:
            self.logger.error(f"Redis get error: {str(e)}")
        
        return None
    
    def put(self, key: str, value: Any, ttl_seconds: int = 3600) -> bool:
        """Put value in distributed cache."""
        if not self.redis_client:
            return False
        
        try:
            serialized_value = pickle.dumps(value)
            self.redis_client.setex(key, ttl_seconds, serialized_value)
            return True
        except Exception as e:
            self.logger.error(f"Redis put error: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from distributed cache."""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            self.logger.error(f"Redis delete error: {str(e)}")
            return False


class CacheManager:
    """Multi-level cache manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize cache levels
        self.l1_cache = LRUCache(
            max_size=config.get('l1_max_size', 1000),
            max_memory_mb=config.get('l1_max_memory_mb', 100)
        )
        
        self.l2_cache = DistributedCache(config.get('redis_url'))
        
        # Cache categories
        self.caches = {
            'rules': LRUCache(max_size=500, max_memory_mb=50),
            'models': LRUCache(max_size=10, max_memory_mb=500),
            'features': LRUCache(max_size=2000, max_memory_mb=200),
            'results': LRUCache(max_size=5000, max_memory_mb=100)
        }
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'puts': 0
        }
        self._stats_lock = threading.Lock()
    
    def get(self, key: str, category: str = 'default') -> Optional[Any]:
        """Get value from appropriate cache level."""
        cache_key = self._generate_cache_key(key, category)
        
        # Try L1 cache first
        cache = self.caches.get(category, self.l1_cache)
        value = cache.get(cache_key)
        
        if value is not None:
            self._record_hit()
            return value
        
        # Try L2 cache (distributed)
        value = self.l2_cache.get(cache_key)
        if value is not None:
            # Promote to L1
            cache.put(cache_key, value)
            self._record_hit()
            return value
        
        self._record_miss()
        return None
    
    def put(self, key: str, value: Any, category: str = 'default', 
            ttl_seconds: int = 3600) -> bool:
        """Put value in appropriate cache levels."""
        cache_key = self._generate_cache_key(key, category)
        
        # Put in L1 cache
        cache = self.caches.get(category, self.l1_cache)
        l1_success = cache.put(cache_key, value)
        
        # Put in L2 cache if available
        l2_success = self.l2_cache.put(cache_key, value, ttl_seconds)
        
        self._record_put()
        return l1_success or l2_success
    
    def delete(self, key: str, category: str = 'default') -> bool:
        """Delete value from all cache levels."""
        cache_key = self._generate_cache_key(key, category)
        
        # Delete from L1
        cache = self.caches.get(category, self.l1_cache)
        cache.delete(cache_key) # Use the new delete method

        # Delete from L2
        self.l2_cache.delete(cache_key)
        
        return True
    
    def clear_category(self, category: str):
        """Clear all entries in a category."""
        if category in self.caches:
            self.caches[category].clear()
    
    def clear_all(self):
        """Clear all caches."""
        for cache in self.caches.values():
            cache.clear()
        self.l1_cache.clear()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._stats_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            stats = {
                'overall': {
                    'hit_rate_percent': hit_rate,
                    'total_requests': total_requests,
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'puts': self.stats['puts']
                },
                'caches': {}
            }
            
            # Get stats for each cache
            for name, cache in self.caches.items():
                stats['caches'][name] = cache.get_stats()
            
            return stats
    
    def optimize_memory_usage(self):
        """Optimize memory usage across caches."""
        memory_percent = psutil.virtual_memory().percent
        
        if memory_percent > 85:
            self.logger.warning(f"High memory usage ({memory_percent}%), optimizing caches")
            
            # Clear least important caches first
            cache_priority = ['features', 'results', 'rules', 'models']
            
            for cache_name in cache_priority:
                if cache_name in self.caches:
                    # Clear half the entries
                    cache = self.caches[cache_name]
                    entries_to_remove = len(cache.cache) // 2
                    
                    with cache._lock:
                        for _ in range(entries_to_remove):
                            if not cache.cache: # Check if cache is empty
                                break
                            _key, removed_entry = cache.cache.popitem(last=False)
                            cache.total_size_bytes -= removed_entry.size_bytes
                
                # Check if memory pressure is relieved
                if psutil.virtual_memory().percent < 80:
                    break
    
    def _generate_cache_key(self, key: str, category: str) -> str:
        """Generate cache key with category prefix."""
        return f"{category}:{hashlib.md5(key.encode()).hexdigest()}"
    
    def _record_hit(self):
        """Record cache hit."""
        with self._stats_lock:
            self.stats['hits'] += 1
    
    def _record_miss(self):
        """Record cache miss."""
        with self._stats_lock:
            self.stats['misses'] += 1
    
    def _record_put(self):
        """Record cache put."""
        with self._stats_lock:
            self.stats['puts'] += 1


# Global cache manager instance
cache_manager = None

def get_cache_manager(config: Dict[str, Any] = None) -> CacheManager:
    """Get global cache manager instance."""
    global cache_manager
    if cache_manager is None and config:
        cache_manager = CacheManager(config)
    return cache_manager