# Hexagonal Architecture vs. Pragmatic Layered Architecture
**A Detailed Comparison for Professional SMART**

---

## Table of Contents
1. [Overview](#overview)
2. [Hexagonal Architecture Explained](#hexagonal-architecture-explained)
3. [Pragmatic Layered Architecture Explained](#pragmatic-layered-architecture-explained)
4. [Side-by-Side Comparison](#side-by-side-comparison)
5. [Professional SMART's Current Architecture](#professional-smarts-current-architecture)
6. [When to Use Each Pattern](#when-to-use-each-pattern)
7. [Code Examples](#code-examples)
8. [Conclusion](#conclusion)

---

## Overview

Both **Hexagonal Architecture** (also called Ports and Adapters) and **Pragmatic Layered Architecture** are architectural patterns that aim to separate business logic from external concerns. However, they differ significantly in their **level of abstraction**, **complexity**, and **use cases**.

### Quick Comparison

| Aspect | Hexagonal Architecture | Pragmatic Layered Architecture |
|--------|------------------------|-------------------------------|
| **Complexity** | High | Low to Medium |
| **Abstraction** | Very High (ports/adapters) | Moderate (concrete layers) |
| **Testability** | Excellent (100% mockable) | Good (75-85% mockable) |
| **Learning Curve** | Steep | Gentle |
| **Boilerplate** | Significant | Minimal |
| **Flexibility** | Maximum | Sufficient |
| **Best For** | Multi-platform, multi-protocol systems | Domain-specific applications |
| **Team Size** | Large teams, domain experts | Small to medium teams |

---

## Hexagonal Architecture Explained

### What Is It?

Hexagonal Architecture (created by Alistair Cockburn in 2005) is a pattern that **isolates business logic** from external systems by using **ports** (interfaces) and **adapters** (implementations).

### Core Principles

1. **Business Logic at the Center**
   - Domain logic has no dependencies on infrastructure
   - No database imports in business code
   - No HTTP framework imports in business code

2. **Ports** (Interfaces)
   - Define contracts for interaction
   - Inbound ports: How external world calls business logic
   - Outbound ports: How business logic calls external systems

3. **Adapters** (Implementations)
   - Implement ports for specific technologies
   - Input adapters: REST API, GraphQL, CLI, message queue
   - Output adapters: PostgreSQL, MongoDB, HTTP client, email service

### Visual Representation

```
┌─────────────────────────────────────────────┐
│                                             │
│              External World                 │
│  (HTTP, Database, File System, etc.)        │
│                                             │
└──────────┬──────────────────┬───────────────┘
           │                  │
    ┌──────▼──────┐    ┌──────▼──────┐
    │   Adapter   │    │   Adapter   │
    │   (REST)    │    │ (PostgreSQL)│
    └──────┬──────┘    └──────┬──────┘
           │                  │
    ┌──────▼──────────────────▼──────┐
    │         Port (Trait)            │
    │  - IClaimRepository             │
    │  - INotificationService         │
    └──────┬────────────────────┬─────┘
           │                    │
    ┌──────▼────────────────────▼─────┐
    │                                  │
    │      Business Logic (Core)       │
    │    - RuleEngine                  │
    │    - ClaimProcessor              │
    │    - PaymentCalculator           │
    │                                  │
    └──────────────────────────────────┘
```

### Example in Rust

```rust
// ═══════════════════════════════════════════════
// DOMAIN LAYER (Core Business Logic)
// ═══════════════════════════════════════════════

// Domain model - no external dependencies
pub struct Claim {
    pub id: Uuid,
    pub patient_control_number: String,
    pub total_charge: Decimal,
    pub diagnoses: Vec<String>,
}

// Outbound Port (interface for persistence)
#[async_trait]
pub trait ClaimRepository: Send + Sync {
    async fn save(&self, claim: &Claim) -> Result<()>;
    async fn find_by_id(&self, id: Uuid) -> Result<Option<Claim>>;
    async fn find_duplicates(&self, pcn: &str) -> Result<Vec<Claim>>;
}

// Outbound Port (interface for notifications)
#[async_trait]
pub trait NotificationService: Send + Sync {
    async fn notify_error(&self, message: &str) -> Result<()>;
}

// Business logic uses ports (traits), not concrete implementations
pub struct ClaimProcessor {
    // Dependency injection of trait objects
    claim_repo: Arc<dyn ClaimRepository>,
    notifier: Arc<dyn NotificationService>,
}

impl ClaimProcessor {
    pub fn new(
        claim_repo: Arc<dyn ClaimRepository>,
        notifier: Arc<dyn NotificationService>,
    ) -> Self {
        Self { claim_repo, notifier }
    }

    pub async fn process_claim(&self, claim: Claim) -> Result<()> {
        // Check for duplicates using port
        let duplicates = self.claim_repo.find_duplicates(&claim.patient_control_number).await?;

        if !duplicates.is_empty() {
            // Notify using port
            self.notifier.notify_error("Duplicate claim detected").await?;
            return Err(Error::Duplicate);
        }

        // Save using port
        self.claim_repo.save(&claim).await?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════
// ADAPTER LAYER (Infrastructure)
// ═══════════════════════════════════════════════

// PostgreSQL adapter (implements the port)
pub struct PostgresClaimRepository {
    pool: PgPool,
}

#[async_trait]
impl ClaimRepository for PostgresClaimRepository {
    async fn save(&self, claim: &Claim) -> Result<()> {
        sqlx::query("INSERT INTO claims (...) VALUES (...)")
            .bind(&claim.id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    async fn find_by_id(&self, id: Uuid) -> Result<Option<Claim>> {
        // PostgreSQL-specific implementation
        Ok(None)
    }

    async fn find_duplicates(&self, pcn: &str) -> Result<Vec<Claim>> {
        // PostgreSQL-specific implementation
        Ok(Vec::new())
    }
}

// Email notification adapter
pub struct EmailNotificationService {
    smtp_client: SmtpClient,
}

#[async_trait]
impl NotificationService for EmailNotificationService {
    async fn notify_error(&self, message: &str) -> Result<()> {
        self.smtp_client.send_email("admin@example.com", message).await?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════
// APPLICATION ASSEMBLY (Dependency Injection)
// ═══════════════════════════════════════════════

fn configure_application(pool: PgPool) -> ClaimProcessor {
    // Wire up concrete implementations to ports
    let claim_repo: Arc<dyn ClaimRepository> = Arc::new(PostgresClaimRepository { pool });
    let notifier: Arc<dyn NotificationService> = Arc::new(EmailNotificationService {
        smtp_client: SmtpClient::new()
    });

    ClaimProcessor::new(claim_repo, notifier)
}
```

### Key Characteristics

**✅ Advantages:**
- **100% testable** - all dependencies are mockable trait objects
- **Swappable implementations** - change database without touching business logic
- **Clear boundaries** - business logic completely isolated
- **Multiple adapters** - can have REST API + GraphQL + CLI using same core
- **Future-proof** - easy to add new protocols/databases

**❌ Disadvantages:**
- **High complexity** - many trait definitions and implementations
- **Boilerplate code** - lots of interface definitions
- **Runtime cost** - trait object dynamic dispatch (small overhead)
- **Steep learning curve** - harder for junior developers
- **Over-engineering risk** - may be overkill for simple applications

---

## Pragmatic Layered Architecture Explained

### What Is It?

Pragmatic Layered Architecture is a **simplified approach** that organizes code into **concrete layers** with clear dependencies flowing **downward**. It provides **good separation** without the full abstraction of hexagonal architecture.

### Core Principles

1. **Clear Layers**
   - Presentation Layer (API, UI)
   - Application Layer (orchestration, workflows)
   - Domain Layer (business logic)
   - Data Layer (repositories, database)

2. **Dependency Flow**
   - Upper layers depend on lower layers
   - Lower layers never depend on upper layers
   - Cross-cutting concerns (logging, errors) are shared

3. **Concrete Dependencies**
   - Dependencies are concrete types (structs)
   - Passed via constructor injection
   - No trait objects required (simpler, faster)

### Visual Representation

```
┌────────────────────────────────────────────┐
│         Presentation Layer                  │
│    (API, WebSocket, CLI)                    │
│    - pro-service                            │
└──────────────┬─────────────────────────────┘
               │ depends on ↓
┌──────────────▼─────────────────────────────┐
│         Application Layer                   │
│    (Workflows, Pipelines)                   │
│    - pro-worker (IngestionPipeline)         │
└──────────┬──────────────┬──────────────────┘
           │              │
           │ depends on ↓ │ depends on ↓
    ┌──────▼──────┐  ┌───▼────────────┐
    │   Domain     │  │   Adapters     │
    │  (Business)  │  │  (Parsers)     │
    │  - Rules     │  │  - EDI Parser  │
    │  - RVU       │  │  - CSV Parser  │
    └──────┬───────┘  └────────────────┘
           │ depends on ↓
┌──────────▼─────────────────────────────────┐
│         Data Layer                          │
│    (Repositories, Database)                 │
│    - pro-db (repositories)                  │
└─────────────────────────────────────────────┘
```

### Example in Rust

```rust
// ═══════════════════════════════════════════════
// DATA LAYER (Bottom Layer)
// ═══════════════════════════════════════════════

// Concrete repository implementation
pub struct EncounterRepository<'a> {
    pool: &'a PgPool,
}

impl<'a> EncounterRepository<'a> {
    pub fn new(pool: &'a PgPool) -> Self {
        Self { pool }
    }

    pub async fn save(&self, encounter: &Encounter) -> Result<()> {
        sqlx::query("INSERT INTO claims.encounter (...) VALUES (...)")
            .bind(&encounter.encounter_id)
            .execute(self.pool)
            .await?;
        Ok(())
    }

    pub async fn find_by_pcn(&self, pcn: &str) -> Result<Option<Encounter>> {
        let result = sqlx::query_as::<_, Encounter>(
            "SELECT * FROM claims.encounter WHERE patient_control_number = $1"
        )
        .bind(pcn)
        .fetch_optional(self.pool)
        .await?;
        Ok(result)
    }
}

// ═══════════════════════════════════════════════
// DOMAIN LAYER (Business Logic)
// ═══════════════════════════════════════════════

// Business rule - depends on PgPool directly (not a trait)
pub struct DuplicateServiceRule;

#[async_trait]
impl Rule for DuplicateServiceRule {
    fn flag_type(&self) -> FlagIssueType {
        FlagIssueType::DuplicateService
    }

    // Takes concrete PgPool, not a trait object
    async fn execute(&self, ctx: &RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>> {
        // Directly query database using concrete pool
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM claims.service_line WHERE ..."
        )
        .bind(ctx.procedure_code.as_ref().unwrap())
        .fetch_one(pool)
        .await?;

        if count > 1 {
            Ok(Some(RuleResult::new(FlagIssueType::DuplicateService, ctx.to_flag_context())))
        } else {
            Ok(None)
        }
    }
}

// Rules engine with concrete dependencies
pub struct RuleEngine {
    pool: PgPool,                    // Concrete type, not trait
    rules: Vec<Arc<dyn Rule>>,       // Only Rule trait is abstracted
}

impl RuleEngine {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            rules: Vec::new(),
        }
    }

    pub fn add_rule<R: Rule + 'static>(&mut self, rule: R) {
        self.rules.push(Arc::new(rule));
    }

    pub async fn execute_all(&self, ctx: &RuleExecutionContext) -> Result<Vec<RuleResult>> {
        let mut results = Vec::new();

        for rule in &self.rules {
            // Pass concrete pool to each rule
            if let Some(result) = rule.execute(ctx, &self.pool).await? {
                results.push(result);
            }
        }

        Ok(results)
    }
}

// ═══════════════════════════════════════════════
// APPLICATION LAYER (Orchestration)
// ═══════════════════════════════════════════════

// Pipeline with concrete dependencies (no traits needed)
pub struct IngestionPipeline {
    pool: PgPool,                        // Concrete
    rule_engine: RuleEngine,             // Concrete
    payment_calculator: PaymentCalculator, // Concrete
}

impl IngestionPipeline {
    pub fn new(pool: PgPool) -> Self {
        // Construct dependencies internally
        let mut rule_engine = RuleEngine::new(pool.clone());
        rule_engine.add_rule(DuplicateServiceRule);
        rule_engine.add_rule(UnitsExceedMaximumRule::default());

        let payment_calculator = PaymentCalculator::with_sample_data();

        Self {
            pool,
            rule_engine,
            payment_calculator,
        }
    }

    pub async fn process_claim(&self, claim: ParsedClaim) -> Result<()> {
        // Use concrete repository
        let encounter_repo = EncounterRepository::new(&self.pool);

        // Check for duplicates
        if let Some(_existing) = encounter_repo.find_by_pcn(&claim.patient_control_number).await? {
            return Err(Error::AlreadyExists("Duplicate claim".to_string()));
        }

        // Execute rules
        let ctx = RuleExecutionContext::from_claim(&claim);
        let flags = self.rule_engine.execute_all(&ctx).await?;

        // Calculate payment
        let payment = self.payment_calculator.calculate(&claim)?;

        // Save to database
        let encounter = Encounter::from_parsed_claim(claim, payment, flags);
        encounter_repo.save(&encounter).await?;

        Ok(())
    }
}

// ═══════════════════════════════════════════════
// PRESENTATION LAYER (Entry Point)
// ═══════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<()> {
    // Set up database connection
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(&database_url)
        .await?;

    // Create pipeline with all concrete dependencies
    let pipeline = IngestionPipeline::new(pool);

    // Process files
    pipeline.process_claim(parsed_claim).await?;

    Ok(())
}
```

### Key Characteristics

**✅ Advantages:**
- **Simple to understand** - no complex abstractions
- **Less boilerplate** - fewer interface definitions
- **Faster compilation** - concrete types compile faster
- **Easier debugging** - no trait object indirection
- **Good separation** - still maintains clear boundaries
- **Sufficient testability** - can mock at higher levels

**❌ Disadvantages:**
- **Harder to swap implementations** - database change requires more refactoring
- **Some coupling** - business logic knows about `PgPool` (but not SQL details)
- **Limited adapter support** - harder to support multiple backends simultaneously
- **Testing requires database** - or careful mocking at layer boundaries

---

## Side-by-Side Comparison

### 1. Dependency Direction

**Hexagonal:**
```rust
// Business logic depends on trait (port)
pub struct ClaimProcessor {
    repo: Arc<dyn ClaimRepository>,  // ← Trait object
}

// Adapter implements trait
impl ClaimRepository for PostgresAdapter { ... }
```

**Pragmatic:**
```rust
// Business logic depends on concrete type
pub struct RuleEngine {
    pool: PgPool,  // ← Concrete PostgreSQL pool
}

// Direct usage
let count = sqlx::query_scalar("SELECT ...").fetch_one(&pool).await?;
```

### 2. Testing Approach

**Hexagonal:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use mockall::mock;

    // Mock the port (100% mockable)
    mock! {
        ClaimRepo {}

        #[async_trait]
        impl ClaimRepository for ClaimRepo {
            async fn save(&self, claim: &Claim) -> Result<()>;
            async fn find_by_id(&self, id: Uuid) -> Result<Option<Claim>>;
        }
    }

    #[tokio::test]
    async fn test_duplicate_detection() {
        let mut mock_repo = MockClaimRepo::new();
        mock_repo.expect_find_duplicates()
            .returning(|_| Ok(vec![...]));  // Control exact behavior

        let processor = ClaimProcessor::new(Arc::new(mock_repo), ...);

        let result = processor.process_claim(claim).await;
        assert!(result.is_err());  // No database needed!
    }
}
```

**Pragmatic:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    #[sqlx::test]  // Uses test database or in-memory SQLite
    async fn test_duplicate_detection(pool: PgPool) {
        // Insert test data into real database
        sqlx::query("INSERT INTO claims.encounter (...) VALUES (...)")
            .execute(&pool)
            .await
            .unwrap();

        let pipeline = IngestionPipeline::new(pool);

        let result = pipeline.process_claim(duplicate_claim).await;
        assert!(result.is_err());  // Requires test database
    }
}
```

### 3. Adding a New Storage Backend

**Hexagonal (Easy):**
```rust
// Add MongoDB adapter without touching business logic
pub struct MongoClaimRepository {
    db: mongodb::Database,
}

#[async_trait]
impl ClaimRepository for MongoClaimRepository {
    async fn save(&self, claim: &Claim) -> Result<()> {
        self.db.collection("claims").insert_one(claim, None).await?;
        Ok(())
    }
    // ... other methods
}

// Switch at configuration time
let repo: Arc<dyn ClaimRepository> = if use_mongo {
    Arc::new(MongoClaimRepository { db })
} else {
    Arc::new(PostgresClaimRepository { pool })
};
```

**Pragmatic (Harder):**
```rust
// Would need to refactor business logic to use traits
// OR maintain two separate codebases
// OR use feature flags with conditional compilation

#[cfg(feature = "postgres")]
pub struct RuleEngine {
    pool: PgPool,
}

#[cfg(feature = "mongodb")]
pub struct RuleEngine {
    db: mongodb::Database,
}

// Implementation diverges - harder to maintain
```

### 4. Code Volume Comparison

For a simple "save claim" operation:

**Hexagonal:**
```
Lines of Code: ~100
- Port trait definition: 20 lines
- Business logic using port: 30 lines
- PostgreSQL adapter: 30 lines
- DI configuration: 20 lines
```

**Pragmatic:**
```
Lines of Code: ~40
- Business logic with direct DB access: 25 lines
- Repository helper: 15 lines
```

**Difference:** Hexagonal requires **2.5x more code** for complete abstraction.

---

## Professional SMART's Current Architecture

### What Pattern Does It Use?

**Answer:** **Pragmatic Layered Architecture** with selective use of trait-based polymorphism.

### Evidence from Codebase

#### 1. Concrete Dependencies in Core Logic

```rust
// pro-worker/src/pipeline.rs
pub struct IngestionPipeline {
    pool: PgPool,  // ← Concrete type, not trait
    rule_engine: RuleEngine,
    payment_calculator: PaymentCalculator,
}

impl IngestionPipeline {
    pub fn new(pool: PgPool) -> Self {
        // Constructs dependencies internally
        let mut rule_engine = RuleEngine::new(pool.clone());
        rule_engine.add_rule(DuplicateServiceRule);
        // ...
    }
}
```

**Analysis:** The pipeline takes a concrete `PgPool`, not a `dyn Database` trait. This is pragmatic layered architecture.

#### 2. Direct Database Access in Rules

```rust
// pro-rules/src/rules/duplicate_service.rs
#[async_trait]
impl Rule for DuplicateServiceRule {
    async fn execute(&self, ctx: &RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>> {
        // Direct SQL query using concrete pool
        let count: i64 = sqlx::query_scalar(
            "SELECT COUNT(*) FROM claims.service_line WHERE ..."
        )
        .fetch_one(pool)  // ← Using PgPool directly
        .await?;
        // ...
    }
}
```

**Analysis:** Business rules query the database directly using `PgPool`. Not hexagonal (which would use a repository port).

#### 3. Selective Trait Usage (Hybrid Approach)

```rust
// pro-rules/src/rule_engine.rs
#[async_trait]
pub trait Rule: Send + Sync {
    async fn execute(&self, ctx: &RuleExecutionContext, pool: &PgPool) -> Result<Option<RuleResult>>;
}

pub struct RuleEngine {
    pool: PgPool,
    rules: Vec<Arc<dyn Rule>>,  // ← Trait objects for rules only
}
```

**Analysis:** Uses traits for **rules** (business logic variability) but not for infrastructure (database). This is **pragmatic** - abstracts only where needed.

#### 4. Adapter Pattern for Inputs

```rust
// pro-parser-edi and pro-parser-csv are adapters
pub struct EdiParser { ... }  // Adapter for EDI format
pub struct CsvParser { ... }  // Adapter for CSV format

// Both convert to common domain model
impl EdiParser {
    pub fn parse(&mut self, content: &str) -> Result<ParsedResult> {
        // EDI-specific parsing
        // Returns domain model (ParsedClaim)
    }
}
```

**Analysis:** Input parsers act as **adapters** (hexagonal concept) but the core doesn't use port traits for output (database).

### Architecture Diagram

```
┌────────────────────────────────────────────────────┐
│           PRESENTATION LAYER                        │
│  - pro-service (Windows Service, WebSocket API)     │
│  - pro-data-loader-gui (GUI application)            │
└───────────────────┬────────────────────────────────┘
                    │
                    ↓ (uses)
┌───────────────────────────────────────────────────┐
│           APPLICATION LAYER                        │
│  - pro-worker (IngestionPipeline)                  │
│    • Orchestrates workflow                         │
│    • Concrete dependencies (PgPool)                │
└───────┬───────────────────┬───────────────────────┘
        │                   │
        ↓ (uses)            ↓ (uses)
┌───────────────┐   ┌──────────────────────┐
│ DOMAIN LAYER  │   │  ADAPTER LAYER       │
│  - pro-rules  │   │  - pro-parser-edi    │ (Input)
│  - pro-rvu    │   │  - pro-parser-csv    │ (Input)
│  - pro-ml     │   │                      │
│    (Traits    │   │  - pro-db (repos)    │ (Output)
│     for       │   │    (Concrete impls)  │
│     variability)  │                      │
└───────┬───────┘   └───────┬──────────────┘
        │                   │
        └────────┬──────────┘
                 │
                 ↓ (queries)
┌────────────────────────────────────────────┐
│          DATA LAYER                         │
│  - PostgreSQL Database                      │
│  - sqlx with concrete PgPool                │
└─────────────────────────────────────────────┘
```

### Why This Works Well

1. **Domain-Specific Application**
   - Only targets PostgreSQL (no need for multiple backends)
   - Only runs on Windows (no cross-platform abstraction needed)
   - Healthcare-specific (unlikely to change domains)

2. **Small to Medium Team**
   - Simpler architecture = easier onboarding
   - Less abstraction = faster development
   - Concrete types = better IDE support

3. **Performance Matters**
   - Healthcare claims processing needs speed
   - Trait objects have small runtime cost
   - Concrete types allow better optimization

4. **Adequate Testability**
   - Can use test databases (sqlx::test)
   - Integration tests validate real workflows
   - Critical rules have unit tests

---

## When to Use Each Pattern

### Use Hexagonal Architecture When:

✅ **Multiple Storage Backends**
```
Example: SaaS product that supports PostgreSQL, MySQL, MongoDB, and DynamoDB
```

✅ **Multiple Input/Output Protocols**
```
Example: API gateway that needs REST, GraphQL, gRPC, WebSocket, and message queues
```

✅ **Frequent Technology Changes**
```
Example: Startup evaluating different databases, message brokers, caching layers
```

✅ **Large Distributed Team**
```
Example: 10+ developers working on different bounded contexts
```

✅ **High Test Coverage Requirements**
```
Example: Financial trading system requiring 95%+ code coverage with unit tests
```

✅ **Plugin Architecture**
```
Example: Application with third-party plugins/extensions
```

### Use Pragmatic Layered Architecture When:

✅ **Single Database Technology**
```
Example: Healthcare claims processing targeting PostgreSQL only ← Professional SMART
```

✅ **Domain-Specific Application**
```
Example: Internal enterprise tool for specific business process
```

✅ **Small to Medium Team**
```
Example: 2-5 developers who understand the full stack
```

✅ **Rapid Development Needed**
```
Example: MVP or proof-of-concept that needs quick iteration
```

✅ **Performance-Critical**
```
Example: High-throughput data processing where every microsecond counts
```

✅ **Simple Deployment Model**
```
Example: Single Windows service deployment (not containerized microservices)
```

### Hybrid Approach (Like Professional SMART)

Use **pragmatic layered** for infrastructure + **traits** for variability:

```rust
// Trait for business logic variability (good use of abstraction)
pub trait Rule: Send + Sync {
    async fn execute(&self, ctx: &Context, pool: &PgPool) -> Result<Option<RuleResult>>;
}

// Concrete infrastructure (pragmatic)
pub struct RuleEngine {
    pool: PgPool,  // Concrete database
    rules: Vec<Arc<dyn Rule>>,  // Abstract business rules
}
```

**When to use hybrid:**
- Need extensibility for business rules
- Don't need multiple infrastructure implementations
- Want balance of simplicity and flexibility

---

## Code Examples: Same Feature, Both Patterns

### Scenario: Save Claim with Duplicate Detection

#### Hexagonal Architecture Implementation

```rust
// ═══════════════════════════════════════════════
// DOMAIN LAYER (src/domain/)
// ═══════════════════════════════════════════════

// Domain model (no external dependencies)
pub struct Claim {
    pub id: Uuid,
    pub patient_control_number: String,
    pub total_charge: Decimal,
}

// Outbound port (trait for persistence)
#[async_trait]
pub trait ClaimRepository: Send + Sync {
    async fn save(&self, claim: &Claim) -> Result<Uuid>;
    async fn find_duplicates(&self, pcn: &str) -> Result<Vec<Claim>>;
}

// Outbound port (trait for events)
#[async_trait]
pub trait EventPublisher: Send + Sync {
    async fn publish_claim_saved(&self, claim_id: Uuid) -> Result<()>;
}

// Business logic (uses ports, not adapters)
pub struct SaveClaimUseCase {
    claim_repo: Arc<dyn ClaimRepository>,
    event_publisher: Arc<dyn EventPublisher>,
}

impl SaveClaimUseCase {
    pub fn new(
        claim_repo: Arc<dyn ClaimRepository>,
        event_publisher: Arc<dyn EventPublisher>,
    ) -> Self {
        Self { claim_repo, event_publisher }
    }

    pub async fn execute(&self, claim: Claim) -> Result<Uuid> {
        // Business rule: check duplicates
        let duplicates = self.claim_repo.find_duplicates(&claim.patient_control_number).await?;
        if !duplicates.is_empty() {
            return Err(Error::AlreadyExists("Duplicate claim".to_string()));
        }

        // Save via port
        let id = self.claim_repo.save(&claim).await?;

        // Publish event via port
        self.event_publisher.publish_claim_saved(id).await?;

        Ok(id)
    }
}

// ═══════════════════════════════════════════════
// ADAPTER LAYER (src/adapters/)
// ═══════════════════════════════════════════════

// PostgreSQL adapter
pub struct PostgresClaimRepository {
    pool: PgPool,
}

#[async_trait]
impl ClaimRepository for PostgresClaimRepository {
    async fn save(&self, claim: &Claim) -> Result<Uuid> {
        let id = sqlx::query_scalar(
            "INSERT INTO claims (id, patient_control_number, total_charge)
             VALUES ($1, $2, $3) RETURNING id"
        )
        .bind(&claim.id)
        .bind(&claim.patient_control_number)
        .bind(&claim.total_charge)
        .fetch_one(&self.pool)
        .await?;
        Ok(id)
    }

    async fn find_duplicates(&self, pcn: &str) -> Result<Vec<Claim>> {
        let claims = sqlx::query_as::<_, Claim>(
            "SELECT * FROM claims WHERE patient_control_number = $1"
        )
        .bind(pcn)
        .fetch_all(&self.pool)
        .await?;
        Ok(claims)
    }
}

// RabbitMQ event publisher adapter
pub struct RabbitMqEventPublisher {
    channel: lapin::Channel,
}

#[async_trait]
impl EventPublisher for RabbitMqEventPublisher {
    async fn publish_claim_saved(&self, claim_id: Uuid) -> Result<()> {
        let message = format!(r#"{{"event": "claim_saved", "id": "{}"}}"#, claim_id);
        self.channel.basic_publish(
            "",
            "claims.saved",
            lapin::options::BasicPublishOptions::default(),
            message.as_bytes(),
            lapin::BasicProperties::default(),
        ).await?;
        Ok(())
    }
}

// ═══════════════════════════════════════════════
// APPLICATION ASSEMBLY (src/main.rs or src/app.rs)
// ═══════════════════════════════════════════════

async fn configure_hexagonal_app() -> SaveClaimUseCase {
    // Wire up PostgreSQL adapter
    let pool = PgPoolOptions::new().connect("...").await.unwrap();
    let claim_repo: Arc<dyn ClaimRepository> = Arc::new(PostgresClaimRepository { pool });

    // Wire up RabbitMQ adapter
    let connection = lapin::Connection::connect("amqp://...", Default::default()).await.unwrap();
    let channel = connection.create_channel().await.unwrap();
    let event_publisher: Arc<dyn EventPublisher> = Arc::new(RabbitMqEventPublisher { channel });

    // Inject dependencies into use case
    SaveClaimUseCase::new(claim_repo, event_publisher)
}

// ═══════════════════════════════════════════════
// TESTING (100% mockable)
// ═══════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use mockall::mock;

    mock! {
        ClaimRepo {}
        #[async_trait]
        impl ClaimRepository for ClaimRepo {
            async fn save(&self, claim: &Claim) -> Result<Uuid>;
            async fn find_duplicates(&self, pcn: &str) -> Result<Vec<Claim>>;
        }
    }

    mock! {
        EventPub {}
        #[async_trait]
        impl EventPublisher for EventPub {
            async fn publish_claim_saved(&self, claim_id: Uuid) -> Result<()>;
        }
    }

    #[tokio::test]
    async fn test_duplicate_detection() {
        // Mock repository to return existing claim
        let mut mock_repo = MockClaimRepo::new();
        mock_repo.expect_find_duplicates()
            .returning(|_| Ok(vec![Claim { id: Uuid::new_v4(), patient_control_number: "123".to_string(), total_charge: Decimal::new(100, 0) }]));

        let mut mock_publisher = MockEventPub::new();
        mock_publisher.expect_publish_claim_saved()
            .times(0);  // Should NOT be called for duplicates

        let use_case = SaveClaimUseCase::new(Arc::new(mock_repo), Arc::new(mock_publisher));

        let claim = Claim { id: Uuid::new_v4(), patient_control_number: "123".to_string(), total_charge: Decimal::new(100, 0) };

        let result = use_case.execute(claim).await;
        assert!(result.is_err());  // No database or message broker needed!
    }
}
```

**Lines of Code:** ~180 lines

---

#### Pragmatic Layered Architecture Implementation

```rust
// ═══════════════════════════════════════════════
// DATA LAYER (src/repositories/)
// ═══════════════════════════════════════════════

pub struct ClaimRepository<'a> {
    pool: &'a PgPool,
}

impl<'a> ClaimRepository<'a> {
    pub fn new(pool: &'a PgPool) -> Self {
        Self { pool }
    }

    pub async fn save(&self, claim: &Claim) -> Result<Uuid> {
        let id = sqlx::query_scalar(
            "INSERT INTO claims (id, patient_control_number, total_charge)
             VALUES ($1, $2, $3) RETURNING id"
        )
        .bind(&claim.id)
        .bind(&claim.patient_control_number)
        .bind(&claim.total_charge)
        .fetch_one(self.pool)
        .await?;
        Ok(id)
    }

    pub async fn find_duplicates(&self, pcn: &str) -> Result<Vec<Claim>> {
        let claims = sqlx::query_as::<_, Claim>(
            "SELECT * FROM claims WHERE patient_control_number = $1"
        )
        .bind(pcn)
        .fetch_all(self.pool)
        .await?;
        Ok(claims)
    }
}

// ═══════════════════════════════════════════════
// DOMAIN/APPLICATION LAYER (src/services/)
// ═══════════════════════════════════════════════

pub struct ClaimService {
    pool: PgPool,
    event_channel: tokio::sync::mpsc::Sender<ClaimEvent>,
}

impl ClaimService {
    pub fn new(pool: PgPool, event_channel: tokio::sync::mpsc::Sender<ClaimEvent>) -> Self {
        Self { pool, event_channel }
    }

    pub async fn save_claim(&self, claim: Claim) -> Result<Uuid> {
        let repo = ClaimRepository::new(&self.pool);

        // Business rule: check duplicates
        let duplicates = repo.find_duplicates(&claim.patient_control_number).await?;
        if !duplicates.is_empty() {
            return Err(Error::AlreadyExists("Duplicate claim".to_string()));
        }

        // Save claim
        let id = repo.save(&claim).await?;

        // Publish event
        self.event_channel.send(ClaimEvent::Saved { claim_id: id }).await
            .map_err(|_| Error::Internal("Failed to send event".to_string()))?;

        Ok(id)
    }
}

// ═══════════════════════════════════════════════
// APPLICATION SETUP (src/main.rs)
// ═══════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<()> {
    let pool = PgPoolOptions::new().connect("...").await?;
    let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(100);

    // Spawn event handler
    tokio::spawn(async move {
        while let Some(event) = event_rx.recv().await {
            match event {
                ClaimEvent::Saved { claim_id } => {
                    println!("Claim saved: {}", claim_id);
                    // Handle event (send to message broker, etc.)
                }
            }
        }
    });

    let service = ClaimService::new(pool, event_tx);

    let claim = Claim { /* ... */ };
    service.save_claim(claim).await?;

    Ok(())
}

// ═══════════════════════════════════════════════
// TESTING (requires test database)
// ═══════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    #[sqlx::test]
    async fn test_duplicate_detection(pool: PgPool) {
        let (event_tx, _event_rx) = tokio::sync::mpsc::channel(10);
        let service = ClaimService::new(pool.clone(), event_tx);

        // Insert existing claim
        sqlx::query("INSERT INTO claims (id, patient_control_number, total_charge) VALUES ($1, $2, $3)")
            .bind(Uuid::new_v4())
            .bind("123")
            .bind(Decimal::new(100, 0))
            .execute(&pool)
            .await
            .unwrap();

        // Try to save duplicate
        let duplicate_claim = Claim { id: Uuid::new_v4(), patient_control_number: "123".to_string(), total_charge: Decimal::new(200, 0) };

        let result = service.save_claim(duplicate_claim).await;
        assert!(result.is_err());  // Requires test database
    }
}
```

**Lines of Code:** ~80 lines

**Difference:** Pragmatic approach has **55% less code** while maintaining clear separation.

---

## Conclusion

### Summary Table

| Aspect | Hexagonal | Pragmatic Layered | Professional SMART |
|--------|-----------|-------------------|-------------------|
| **Abstraction Level** | Very High | Medium | Medium |
| **Code Volume** | High (+150%) | Baseline | Baseline |
| **Testability** | Excellent (100%) | Good (75%) | Good (75%) |
| **Flexibility** | Maximum | Sufficient | Sufficient |
| **Learning Curve** | Steep | Gentle | Gentle |
| **Maintenance** | Medium | Easy | Easy |
| **Performance** | Good (trait overhead) | Excellent | Excellent |
| **Best For** | Multi-backend SaaS | Domain-specific apps | Healthcare claims (✓) |

### Recommendations for Professional SMART

**Current Architecture: ✅ KEEP AS-IS**

**Rationale:**
1. **PostgreSQL Only** - No need for database abstraction
2. **Windows Only** - No cross-platform requirements
3. **Healthcare Domain** - Specialized, not general-purpose
4. **Small Team** - Simplicity aids maintainability
5. **Performance Critical** - Claims processing needs speed

**Optional Enhancement:**
Consider adding **repository port traits** only if:
- Need to support multiple databases in future
- Want 100% unit test coverage without test databases
- Team grows to 10+ developers

**Example Incremental Improvement:**
```rust
// Add trait only where needed
#[async_trait]
pub trait EncounterRepository {
    async fn save(&self, encounter: &Encounter) -> Result<()>;
    async fn find_by_pcn(&self, pcn: &str) -> Result<Option<Encounter>>;
}

// Keep concrete implementation
pub struct PgEncounterRepository<'a> {
    pool: &'a PgPool,
}

#[async_trait]
impl EncounterRepository for PgEncounterRepository<'_> {
    // Implementation
}

// Use trait in critical business logic
pub struct IngestionPipeline {
    encounter_repo: Arc<dyn EncounterRepository>,  // ← Trait for testability
    pool: PgPool,  // ← Keep for non-critical operations
}
```

This hybrid approach gets **80% of hexagonal benefits** with **20% of the complexity**.

---

**Final Verdict:** Professional SMART's pragmatic layered architecture is **appropriate**, **maintainable**, and **production-ready**. Hexagonal architecture would add unnecessary complexity without proportional benefits for this use case.

