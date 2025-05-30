# src/async_processing/async_processor.py
"""
Asynchronous Processing System for High-Throughput Claims Validation
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
import aiohttp
import asyncpg
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import queue
import signal
import weakref


@dataclass
class ProcessingTask:
    """Represents a processing task."""
    task_id: str
    claim_data: Dict[str, Any]
    priority: int = 0
    created_time: float = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = time.time()


class AsyncDatabasePool:
    """Async database connection pool manager."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.pool = None
        self._pool_lock = asyncio.Lock()
    
    async def initialize(self):
        """Initialize async database pool."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config['host'],
                port=self.config['port'],
                database=self.config['database'],
                user=self.config['user'],
                password=self.config['password'],
                min_size=self.config.get('min_connections', 5),
                max_size=self.config.get('max_connections', 20),
                command_timeout=60,
                server_settings={
                    'jit': 'off'  # Optimize for short queries
                }
            )
            self.logger.info("Async database pool initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize async pool: {str(e)}")
            raise
    
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute async query."""
        async with self.pool.acquire() as conn:
            try:
                rows = await conn.fetch(query, *args)
                return [dict(row) for row in rows]
            except Exception as e:
                self.logger.error(f"Async query error: {str(e)}")
                raise
    
    async def execute_batch(self, query: str, args_list: List[tuple]) -> bool:
        """Execute batch queries."""
        async with self.pool.acquire() as conn:
            try:
                await conn.executemany(query, args_list)
                return True
            except Exception as e:
                self.logger.error(f"Async batch error: {str(e)}")
                return False
    
    async def close(self):
        """Close database pool."""
        if self.pool:
            await self.pool.close()


class AsyncTaskQueue:
    """High-performance async task queue with priorities."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.high_priority = asyncio.Queue(maxsize=max_size // 3)
        self.normal_priority = asyncio.Queue(maxsize=max_size // 3)
        self.low_priority = asyncio.Queue(maxsize=max_size // 3)
        self.total_size = 0
        self._size_lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
    
    async def put(self, task: ProcessingTask):
        """Add task to appropriate priority queue."""
        async with self._size_lock:
            if self.total_size >= self.max_size:
                raise queue.Full("Task queue is full")
            
            if task.priority >= 8:
                await self.high_priority.put(task)
            elif task.priority >= 5:
                await self.normal_priority.put(task)
            else:
                await self.low_priority.put(task)
            
            self.total_size += 1
    
    async def get(self) -> ProcessingTask:
        """Get next task based on priority."""
        # Try high priority first
        try:
            task = self.high_priority.get_nowait()
            await self._decrement_size()
            return task
        except asyncio.QueueEmpty:
            pass
        
        # Try normal priority
        try:
            task = self.normal_priority.get_nowait()
            await self._decrement_size()
            return task
        except asyncio.QueueEmpty:
            pass
        
        # Wait for any priority
        queues = [self.high_priority, self.normal_priority, self.low_priority]
        done, pending = await asyncio.wait(
            [q.get() for q in queues],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel pending tasks
        for p in pending:
            p.cancel()
        
        task = done.pop().result()
        await self._decrement_size()
        return task
    
    async def _decrement_size(self):
        """Decrement total size counter."""
        async with self._size_lock:
            self.total_size = max(0, self.total_size - 1)
    
    def qsize(self) -> int:
        """Get current queue size."""
        return self.total_size


class AsyncClaimProcessor:
    """Async claim processor with advanced features."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Processing configuration
        self.max_concurrent_workers = config.get('max_concurrent_workers', 20)
        self.batch_size = config.get('batch_size', 100)
        self.processing_timeout = config.get('processing_timeout', 30)
        
        # Components
        self.db_pool = AsyncDatabasePool(config.get('database', {}))
        self.task_queue = AsyncTaskQueue(config.get('queue_size', 10000))
        
        # State management
        self.workers = []
        self.running = False
        self.stats = {
            'processed': 0,
            'errors': 0,
            'start_time': None,
            'processing_times': []
        }
        
        # Graceful shutdown
        self.shutdown_event = asyncio.Event()
        self._setup_signal_handlers()
    
    async def initialize(self):
        """Initialize async processor."""
        await self.db_pool.initialize()
        self.logger.info("Async claim processor initialized")
    
    async def start_processing(self):
        """Start async processing workers."""
        if self.running:
            return
        
        self.running = True
        self.stats['start_time'] = time.time()
        
        # Start worker coroutines
        for i in range(self.max_concurrent_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.workers.append(worker)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitor_progress())
        self.workers.append(monitor_task)
        
        self.logger.info(f"Started {self.max_concurrent_workers} async workers")
    
    async def stop_processing(self):
        """Stop processing gracefully."""
        self.running = False
        self.shutdown_event.set()
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Close database pool
        await self.db_pool.close()
        
        self.logger.info("Async processing stopped")
    
    async def submit_claims(self, claims: List[Dict[str, Any]], priority: int = 5):
        """Submit claims for async processing."""
        tasks_submitted = 0
        
        for claim in claims:
            try:
                task = ProcessingTask(
                    task_id=claim.get('claim_id', f'task-{time.time()}'),
                    claim_data=claim,
                    priority=priority
                )
                await self.task_queue.put(task)
                tasks_submitted += 1
            except queue.Full:
                self.logger.warning("Task queue is full, dropping claim")
                break
        
        self.logger.info(f"Submitted {tasks_submitted} claims for processing")
        return tasks_submitted
    
    async def _worker_loop(self, worker_id: str):
        """Main worker processing loop."""
        self.logger.info(f"Worker {worker_id} started")
        
        while self.running and not self.shutdown_event.is_set():
            try:
                # Get next task with timeout
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=5.0
                )
                
                # Process task
                await self._process_task(task, worker_id)
                
            except asyncio.TimeoutError:
                continue  # Check shutdown condition
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {str(e)}")
                await asyncio.sleep(1)  # Brief pause on error
        
        self.logger.info(f"Worker {worker_id} stopped")
    
    async def _process_task(self, task: ProcessingTask, worker_id: str):
        """Process individual task."""
        start_time = time.time()
        
        try:
            # Validate claim asynchronously
            result = await self._validate_claim_async(task.claim_data)
            
            # Store result asynchronously
            await self._store_result_async(result)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['processed'] += 1
            self.stats['processing_times'].append(processing_time)
            
            # Keep only recent processing times for memory efficiency
            if len(self.stats['processing_times']) > 1000:
                self.stats['processing_times'] = self.stats['processing_times'][-1000:]
            
        except Exception as e:
            self.logger.error(f"Task processing error: {str(e)}")
            
            # Retry logic
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                await self.task_queue.put(task)
                self.logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count})")
            else:
                self.stats['errors'] += 1
                self.logger.error(f"Task {task.task_id} failed after {task.max_retries} retries")
    
    async def _validate_claim_async(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Async claim validation."""
        # This would integrate with your existing validator
        # For now, simulate async validation
        await asyncio.sleep(0.01)  # Simulate processing time
        
        return {
            'claim_id': claim.get('claim_id'),
            'validation_status': 'COMPLETED',
            'predicted_filters': [],
            'validation_results': [],
            'processing_time': 0.01
        }
    
    async def _store_result_async(self, result: Dict[str, Any]):
        """Store validation result asynchronously."""
        query = """
        INSERT INTO ValidationResults 
        (claim_id, validation_status, predicted_filters, validation_details, processing_time)
        VALUES ($1, $2, $3, $4, $5)
        """
        
        await self.db_pool.execute_query(
            query,
            result['claim_id'],
            result['validation_status'],
            str(result['predicted_filters']),
            str(result['validation_results']),
            result['processing_time']
        )
    
    async def _monitor_progress(self):
        """Monitor processing progress."""
        last_processed = 0
        
        while self.running and not self.shutdown_event.is_set():
            await asyncio.sleep(30)  # Monitor every 30 seconds
            
            current_processed = self.stats['processed']
            rate = (current_processed - last_processed) / 30  # Claims per second
            
            if self.stats['processing_times']:
                avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
            else:
                avg_time = 0
            
            self.logger.info(
                f"Processing stats: {current_processed} total, "
                f"{rate:.1f} claims/sec, {avg_time:.3f}s avg time, "
                f"Queue size: {self.task_queue.qsize()}, Errors: {self.stats['errors']}"
            )
            
            last_processed = current_processed
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown")
            asyncio.create_task(self.stop_processing())
        
        try:
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        except Exception:
            pass  # Signal handling might not work in all environments
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        rate = self.stats['processed'] / uptime if uptime > 0 else 0
        
        return {
            'processed_claims': self.stats['processed'],
            'error_count': self.stats['errors'],
            'processing_rate_per_second': rate,
            'uptime_seconds': uptime,
            'queue_size': self.task_queue.qsize(),
            'active_workers': len([w for w in self.workers if not w.done()]),
            'average_processing_time': (
                sum(self.stats['processing_times']) / len(self.stats['processing_times'])
                if self.stats['processing_times'] else 0
            )
        }


class AsyncValidationPipeline:
    """Complete async validation pipeline with streaming."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.processor = AsyncClaimProcessor(config)
        
        # Streaming configuration
        self.stream_enabled = config.get('enable_streaming', False)
        self.kafka_config = config.get('kafka', {})
        
        # Pipeline stages
        self.stages = [
            self._preprocess_stage,
            self._validation_stage,
            self._postprocess_stage
        ]
    
    async def initialize(self):
        """Initialize the complete pipeline."""
        await self.processor.initialize()
        
        if self.stream_enabled:
            await self._initialize_streaming()
        
        self.logger.info("Async validation pipeline initialized")
    
    async def start(self):
        """Start the validation pipeline."""
        await self.processor.start_processing()
        
        if self.stream_enabled:
            await self._start_streaming()
        
        self.logger.info("Async validation pipeline started")
    
    async def stop(self):
        """Stop the validation pipeline."""
        await self.processor.stop_processing()
        
        if self.stream_enabled:
            await self._stop_streaming()
        
        self.logger.info("Async validation pipeline stopped")
    
    async def process_claims_stream(self, claims_generator):
        """Process claims from a generator/stream."""
        batch = []
        
        async for claim in claims_generator:
            batch.append(claim)
            
            if len(batch) >= self.processor.batch_size:
                await self.processor.submit_claims(batch)
                batch = []
        
        # Process remaining claims
        if batch:
            await self.processor.submit_claims(batch)
    
    async def _preprocess_stage(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocessing stage."""
        # Add preprocessing logic here
        # Example: data cleaning, normalization, enrichment
        
        # Simulate async preprocessing
        await asyncio.sleep(0.001)
        
        # Add metadata
        claim['preprocessed_at'] = time.time()
        claim['pipeline_stage'] = 'preprocessing'
        
        return claim
    
    async def _validation_stage(self, claim: Dict[str, Any]) -> Dict[str, Any]:
        """Main validation stage."""
        # This would call your existing validation logic
        # But wrapped in async functions
        
        result = await self.processor._validate_claim_async(claim)
        result['pipeline_stage'] = 'validation'
        
        return result
    
    async def _postprocess_stage(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Postprocessing stage."""
        # Add postprocessing logic here
        # Example: result formatting, notifications, reporting
        
        await asyncio.sleep(0.001)
        
        result['postprocessed_at'] = time.time()
        result['pipeline_stage'] = 'postprocessing'
        
        return result
    
    async def _initialize_streaming(self):
        """Initialize Kafka streaming (optional)."""
        try:
            # This would initialize Kafka consumers/producers
            # Implementation depends on your specific streaming needs
            self.logger.info("Streaming initialized (placeholder)")
        except Exception as e:
            self.logger.error(f"Streaming initialization failed: {str(e)}")
    
    async def _start_streaming(self):
        """Start streaming consumers."""
        self.logger.info("Streaming started (placeholder)")
    
    async def _stop_streaming(self):
        """Stop streaming consumers."""
        self.logger.info("Streaming stopped (placeholder)")


# Usage example and factory functions
class AsyncProcessorFactory:
    """Factory for creating async processors with different configurations."""
    
    @staticmethod
    def create_high_throughput_processor(config: Dict[str, Any]) -> AsyncClaimProcessor:
        """Create processor optimized for high throughput."""
        high_throughput_config = {
            **config,
            'max_concurrent_workers': 50,
            'batch_size': 200,
            'queue_size': 20000,
            'processing_timeout': 60
        }
        return AsyncClaimProcessor(high_throughput_config)
    
    @staticmethod
    def create_low_latency_processor(config: Dict[str, Any]) -> AsyncClaimProcessor:
        """Create processor optimized for low latency."""
        low_latency_config = {
            **config,
            'max_concurrent_workers': 10,
            'batch_size': 50,
            'queue_size': 5000,
            'processing_timeout': 10
        }
        return AsyncClaimProcessor(low_latency_config)
    
    @staticmethod
    def create_memory_optimized_processor(config: Dict[str, Any]) -> AsyncClaimProcessor:
        """Create processor optimized for memory usage."""
        memory_config = {
            **config,
            'max_concurrent_workers': 20,
            'batch_size': 100,
            'queue_size': 5000,
            'processing_timeout': 30
        }
        return AsyncClaimProcessor(memory_config)


# Example usage
async def main():
    """Example usage of async processing system."""
    config = {
        'database': {
            'host': 'localhost',
            'port': 5432,
            'database': 'claims_processing',
            'user': 'postgres',
            'password': 'admin',
            'min_connections': 5,
            'max_connections': 20
        },
        'max_concurrent_workers': 20,
        'batch_size': 100,
        'queue_size': 10000
    }
    
    # Create and initialize processor
    processor = AsyncClaimProcessor(config)
    await processor.initialize()
    
    # Start processing
    await processor.start_processing()
    
    # Submit some test claims
    test_claims = [
        {'claim_id': f'test-{i}', 'patient_age': 25 + i}
        for i in range(1000)
    ]
    
    await processor.submit_claims(test_claims)
    
    # Let it process for a while
    await asyncio.sleep(60)
    
    # Get stats
    stats = processor.get_stats()
    print(f"Processing stats: {stats}")
    
    # Stop processing
    await processor.stop_processing()


if __name__ == "__main__":
    asyncio.run(main())