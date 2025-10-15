# Software Requirements Document — Professional SMART® (Rust + PostgreSQL)

## 1) Purpose & Background
Build a secure, scalable system to improve coding accuracy, enable retrospective audits, track denials, and provide BI/benchmarking dashboards for professional claims (837p / billed charge report). Key business drivers include accuracy, operational improvement, financial performance, and education workflows (see slides “Problem Statement/Benefits” and “Key Features”).  Process 10,000 claims / 15 seconds as its ceiling.

## 2) Scope
- **In-scope (Phase 1):** Retrospective audits & auditor worksheets; import + screening + flagging; management/practice dashboards; coding accuracy metrics; RVU reimbursement estimation based on CMS factors; denials trending (initial).  
- **Out of scope (Phase 1):** Full BI suite externalization, heat-map experiments, RAF module (planned), advanced benchmarking exchanges (Phase 2).
- **Process 837 professional claims data
- **Store the data in a postgresql database using best practices
- **Claims data will run through a customized rules engine that will capture:
Issue Category​
Issue Description​
COD: Bundled Service/Procedure​
Documentation indicates service/procedure billed was performed but is part of, or bundled into, another service​
COD: Incorrect Procedure Code​
Documentation supports a different service/procedure than was billed​
COD: Missed Charge​
Documentation supports a service/procedure that was not billed​
COD: Time Not Documented​
Documentation does not include time and the service billed was a time-based service (non-E/M)​
DOC: Missing Documentation​
No documentation found / provided for service billed​
DOC: Limited/Insufficient Documentation​
Documentation is not sufficient to support the billed service (not including missing signature)​
EMO: E/M Over - One Level​
Documentation supports an E/M code one level lower than was billed​
EMO: E/M Over - Two Levels or More​
Documentation supports an E/M code two or more levels lower than was billed​
EMU: E/M Under - One Level​
Documentation supports an E/M code one level higher than was billed​
EMU: E/M Under - Two Levels or More​
Documentation supports an E/M code two or more levels higher than was billed​
EMI: E/M Incorrect Category​
Documentation supports a different E/M category than was billed​
EMT: E/M Time - Not Documented​
Documentation does not include time and the service billed was a time-based service or coded based on time​
MOD: Incorrect​
Modifier reported is incorrect (a different modifier is required based on the documentation)​
MOD: Missing​
Modifier is required for the billed service but was not reported​
MOD: Unnecessary​
Modifier reported is not required for the procedure(s) billed​
OTH: Incorrect Provider​
Documentation indicates a different provider rendered the service than was billed​
OTH: Incorrect Date of Service​
Documentation indicates the service was performed on a different date of service than was billed​
OTH: Missing Provider Signature​
Documentation is missing the provider's signature​
QTY: Fewer Units Supported​
Documentation supports fewer units than were billed​
QTY: More Units Supported​
Documentation supports more units than were billed​
SUP: Incident to / Split Shared Requirements Not Met​
Documentation does not support incident to / split shared billing requirements​
SUP: Teaching Physician Guideline Requirements Not Met​
Documentation does not support teaching physician guidelines billing requirements​
SUP: Supervision Requirements Not Met​
Documentation does not support required level of supervision for service type provided (e.g., imaging, lab)​
Issue Category​
Issue Description​
Additional Diagnosis​
Additional ICD-10-CM code(s) are documented but were not reported​
Documentation​
Documentation does not support the reported diagnosis code(s)​
Incorrect​
ICD-10-CM code(s) reported are not correct based upon documentation​
Specificity​
Documentation supports a more specific diagnosis code(s) than that reported​

## 3) Users & Roles
- **Auditor:** Executes retrospective reviews, proposes changes, runs reports, schedules audits.  
- **Manager:** Monitors queues, status, workload, KPIs on dashboards.  
- **Coder/Provider:** Subject of accuracy metrics & education workflows.  
- **Analyst/Finance:** Tracks charges vs reimbursement, denials, benchmarking.  

## 4) Core Functional Requirements

### 4.1 Data Ingestion & Validation
1. **Input formats:**  
   - 837p professional claims and/or billed charge reports with encounter detail; CMS-1500 mapping support for 837p fields.  
2. **Validation & De-dup:**  
   - Validate required fields; reject malformed/duplicate encounters; produce data validation reports.  
3. **Exclusion & Exemptions:**  
   - Apply facility/client-defined exemptions; exempt cases load without flags.  
4. **Throughput:**  
   - Handle high-volume, low-charge professional claims; assess storage & scalability.  

### 4.2 Screening & Flagging
5. **Flag engine:**  
   - Run non-exempt encounters through rules to assign **E/M**, **CPT**, **Modifier**, **Dx** flags (post/pre-bill variants).  
6. **Pending workflow:**  
   - Add flagged cases to pending list; pass clean cases to reporting store.  

### 4.3 Retrospective Audits & Evaluations
7. **Auditor worksheets:**  
   - Support specialty-specific audits, targeted by risk areas; track proposed coding changes, outcomes, and education targets.  
8. **Scheduling & alerts:**  
   - Configure audits at intervals or on thresholds; leverage schedule + alert UI/logic.  

### 4.4 Dashboards & Reporting
9. **Management dashboard:**  
   - Status categories (Unassigned, Active, Awaiting Change, Forwarded, Hold); filter by Encounter Date; show E/M, CPT, Modifier, Dx flag categories.  
10. **Coding accuracy:**  
   - Accuracy rates for E/M, CPT, Modifier, Dx by date range, clinical dept, billing provider/coder; detailed coder/provider tables with revision counts.  
11. **Practice dashboards:**  
   - E/M distribution vs benchmarks, CPT usage patterns, RVUs by month/provider; actual vs target wRVUs; new patient visit trends.  
12. **Flags analytics:**  
   - Pre-bill/post-bill flag distribution; flag rates/change rates (where applicable); top/bottom flags by provider/coder.  
13. **Denials trending (835):**  
   - Trend denials by root cause (e.g., registration/eligibility, invalid claim data, authorization, medical necessity, coding); visualize top drivers.  

### 4.5 Reimbursement Estimation
14. **RVU-based estimate:**  
   - Use national RVUs + GPCI + 2024 PFS **conversion factor $33.2875** to estimate reimbursement; respect facility vs non-facility practice expense.  

## 5) Non-Functional Requirements
- **Performance:** p95 < 300 ms for dashboard queries with indexed filters; batch ingest ≥ 1M encounters/day on provisioned hardware.  
- **Scalability:** Horizontal scale for API and workers; partitioning/archiving for fact tables.  
- **Security & Compliance:** PHI handling, RBAC/ABAC, audit trails, encryption at rest (Postgres TDE or disk), TLS in transit.  
- **Observability:** Structured tracing, metrics, and health checks; import & ruleset run logs.  
- **Data Quality:** Validation reports; lineage of changes from audits/worksheet actions.

## 6) System Architecture (Rust + PostgreSQL)

### 6.1 Services
- **Gateway/API:** Rust (`axum` preferred) with `tokio`; JWT/OAuth2; rate limits.  
- **Ingestion Worker:** Streams 837p/CSV; validates, de-dups, stages; pushes to screening.  CSV will need to be custom with headers
- **Flag Engine:** Rust rules service; deterministic, unit-tested; emits flag assignments & reasons.  
- **Audit Service:** Manages evaluations, schedules, notifications, proposed changes.  
- **Analytics API:** Read-optimized endpoints for dashboards & exports.

### 6.2 Tech Stack
- **Rust crates:** `axum`, `serde`, `sqlx` (compile-time checked queries), `uuid`, `time`, `tracing`, `thiserror`, `tokio`, `utoipa`/`okapi` (OpenAPI), `jsonwebtoken` or `oauth2`, `lettre` (email), `chrono-tz`.  
- **DB:** PostgreSQL 15/16; `sqlx::Pool`; migrations via `sqlx migrate`.  
- **Extensions:** `uuid-ossp` (or app-generated UUID/ULID), `citext`, `pg_trgm`, `pgcrypto`.  
- **Storage:** Raw file landing zone (object store) + staged tables; row-level lineage.

## 7) Data Model (initial)

### 7.1 Core Entities
- **encounter**(...)
- **flag**(...)
- **audit**(...)
- **audit_sample**(...)
- **coder_provider**(...)
- **accuracy_metric_daily**(...)
- **denial_event**(...)
- **rvu_reference**(...)
- **schedule**(...)

### 7.2 Indices & Policies
- B-tree on `(dos_ts)`, `(provider_id, dos_ts)`, `(status, dos_ts)`; GIN on flags `(encounter_id, category, code)`; partial index for `is_active=true`.  
- Optional **RLS** by organization/tenant if multi-tenant.

### 7.3 Reimbursement Formula
```
payment = (work_rvu*gpci_work + pe_rvu*gpci_pe + mp_rvu*gpci_mp) * conversion_factor
```

## 8) APIs (REST/JSON)

### 8.1 Ingestion
- `POST /v1/import/encounters` — upload 837p/CSV batch (signed URL or multipart).  
- `GET /v1/import/batches/{id}` — validation results, rejected duplicates/exclusions.

### 8.2 Screening & Flags
- `POST /v1/flags/run?batch_id=…` — execute flag engine for staged encounters.  
- `GET /v1/encounters/{id}/flags` — list flags; `PATCH` to resolve/override with reason.

### 8.3 Audits & Evaluations
- `POST /v1/audits` — create audit (topic/scope/sampling).  
- `POST /v1/audits/{id}/schedule` — attach RRULE; threshold triggers.  
- `GET /v1/audits/{id}/samples` — pull sample; `PATCH` proposed changes/outcomes.

### 8.4 Dashboards & Analytics
- `GET /v1/dash/management`  
- `GET /v1/dash/coding-accuracy`  
- `GET /v1/dash/practice`  
- `GET /v1/dash/flags`  
- `GET /v1/dash/denials`  

### 8.5 Reference & Estimation
- `GET /v1/reference/rvu/{cpt}`  
- `POST /v1/estimate/payment`  

## 9) Rules & Calculations
### 9.1 Flag Rules (high level)
- **E/M**, **CPT**, **Modifier**, **Dx**.

### 9.2 Accuracy Metrics
- accuracy = 1 − (approved revisions / reviewed items).

## 10) Scheduling & Notifications
- RRULE-based schedules; threshold triggers (e.g., flag rate > X%).  

## 11) Security, Privacy, Compliance
- HIPAA-aligned; encryption, RBAC, audit logs.

## 12) Data Retention & Archival
- Hot store for 24 months; archive thereafter.

## 13) Testing & QA
- Unit, integration, data QA, performance.

## 14) Deployment & Operations
- Containerized services, CI/CD, blue/green or canary deployments, observability, runbooks.


Rather than a the Mgt Dashboard we have for IP it would be a Dashboard of Reviews in Progress -  retro auditing tool for internal and for clients - Reviews in Progress and the Historical Reviews at a glance with results 

--- Audits based on Coder, Providers, Provider Group/Specialty - audits need to be able to be set-up in multiple ways - also being able to do Random samples  

----When selecting cases can the criteria sued when selecting the cases be append to the cases so the reviewer can see? 

----have the ability for a secondary review and acceptance of the change - similar to Evaluations in legacy or WQ in SMART6 

----maybe reviewing Evals in legacy to see what there would/could work and having the ability once they finalize for the next case to auto populate and for them to be able to select from a list  

-In the worksheets having the DX and Procs side by side  

-Look at how to incorporate the scoring into the worksheets 

-Assessment WQs 

-Actions - Change, No Change, Hold 

-If the reviewers have questions they right now will email Chandra or Michele - can we have the ability for them to fwd case to them - and can we send out of the system an email with a hyper link to the person/person(s) they fwd to so they can come into the
system (similar to the coder module)- we should have this enhancement for IP - add to backlog   

-Actions at code level with notes required - at the code level will allow the system to weight the cases automatically and display  

-for Secondary reviewers - similar to the coder module with some additional features - ability to agree and disagree with their recommendations and also edit - they do provide reviewers with feedback so we will need to think thru how we can log and provide
info back or does it just go back to the reviewers to make any of the rec updates/changes - can we have an option we select per audit whether we will be sending back to the reviewer to update OR have the secondary review changing the reviews and finalizing
- similar to whether the coder module will be used - if the secondary reviewers are going to edit the changes and not send back to the reviewer to update the case can we have something similar to the coder 'acknowledge' functionality where to case gets fed
back to the Reviewers list for them to review and acknowledge 

  

For Reporting: 

-Reviewers - need to track their results and be able to see accts per day - similar to the Legacy IP Reviewer Report 





Fields
Only looks up what the case looks like when you are running it
Notes
Delivery Team Comments
Patient Control No
Y
supported by Account Number
Med Rec No
Y
supported by Medical Record Number
Date of Service
Y
supported by Date of Encounter
Birth Date
Birth Date
Birth Date
Case Status
Y - Define
AP 1/23 - never discussed how these will be defined in the NextGen prod
New
Case ID
NEW 
New
Case Created Date 
NEW 
Is it still called the same?
New
Change Indicator 
Y
New, DB Schema change needed (revision reason)
Change Indicator Category
Y - implement reasons from our CRWs
Coder ID
Y
Supported by Coder
Coder Group
Y
New
Coding Date
Y (could also use to pull analytics for how soon cases coded that were included w/in the random sample selected)
New
EHR
NEW 
Have as column option
New
Facility Name
ADD sooner than later since we cannot select the facilities we are searching like we do in on prem SMART
Have as column option
New
Facility ID
NEW 
New
Organization 
NEW 
New
Region
NEW - ADD sooner than later since we cannot select the facilities we are searching like we do in on prem SMART
New
Race
NEW 
New
Gender
NEW 
New
Financial Class
Y
Supported
Flag Group - Any
Y
New
Flag Group - Any - First Version
N - will have to inlcude in the future. not the top priority
Flag Group - Primary
N - will have to inlcude in the future. not the top priority
Flag Group - Any - First Version
N - will have to inlcude in the future. not the top priority
Flag Number - Any
Y
Supported
Flag Number - Any - First Version
N - will have to inlcude in the future. not the top priority
Flag Number - Primary
N - will have to inlcude in the future. not the top priority
Flag Number - Primary - First Version
N - will have to inlcude in the future. not the top priority
Flag Cateogry - Any
Y
New
Flag Cateogry - Primary
N - will have to inlcude in the future. not the top priority
Grouper Version
N
CPT/HCPCS Codes - Any
Y
ICD DX Codes - Any
Y
New (Px or Sdx)
ICD DX Codes - Other
N
New (Sdx)
ICD DX Codes - Other - First Version
N - will have to inlcude in the future. not the top priority
ICD DX Codes - Principal
Y
New
ICD DX Codes - Principal - First Version
N - will have to inlcude in the future. not the top priority
ICD DX Codes, HCC Category
Y - label  dif
New (Dx table HCC col)
ICD DX Codes, HCC Indicator
Y
New
ICD DX Codes, Single CC
N
New
ICD DX Codes, Single MCC
N
New
ICD Dx ROM
N - will have to inlcude in the future. not the top priority
ICD Dx SOI
N - will have to inlcude in the future. not the top priority
Import Date
Y - AP 1/23 was not included yet bc they built the critieria based on the export column tab and not the criteria tab
New
Is On Assessment/in a WQ
Y
Or was part of a prior assessment
Is On List
N
Notes
Y (might want to pull reports on notes included w/in an assessment)
Option Field 1
N
Supported
Option Field 2
N
Supported
Option Field 3
New to NextGen since we can now take 4 Optional Fields
New
Option Field 4
New to NextGen since we can now take 4 Optional Fields
New
Patient Sex
Y
Payer
Y
Supported
Physician - Attending
Y
Supported by Attending Physician ID
Physician Group - Attending
?
Physician - Procedure
Y
New (in proc table)
Physician Group - Procedre
?
PPS Type
N
RAC Focus 
N
Random Sample - Auto
Y
Random Sample - Size
***Add for use case when we have cases to put on an evaluation
Rescreen Date
N
Revenue Change
Y
Review Date - Any
Y - to start
New
Review Date - First
Y
New
Review Date - Latest
for later consideration
Reviewer - Any
Y - to start
Supported
Reviewer Group - Any
Y - to start
New
Reviewer - First
Y
New
Reviewer Group - First
Y
New
Reviewer - Latest
for later consideration
Reviewer Group - Latest
for later consideration
Service
Y
New (ehr service table)
Total Charges
Y
supported
Updated By
Last Person who touched the account 
New, DB Schema change needed
Work Queue State
Should be covered in Cases State/Status but we need to understand how that will work in the new product or define for the new product 
may require additional fields
New, DB Schema change needed
Soft Delete
Already discussed
PWC Admin Only and Admin
Upload Log
include UploadString, caseID, outputdate, updateddate, updatedby, import configuration id (search on import configuration name to get id?)
Multi-field search
PWC Admin Only
Coder Communication
PWC Admin Only



Fields/Category
Operator
Patient Control No
same as account
Admit Source - Facility
[=, <>, in, not in]
Admit Type - Facility
[=, <>, in, not in]
Case State
[=, <>, in, not in]
Case ID
what is this?  how is it different 
Case Created Date 
[=, <>, >, >=, <, <=, between,latest]
Case Created Date - Cassandra
[=, <>, >, >=, <, <=, between,latest]
Change Indicator 
[=, <>, in, not in]
Coder Group
[=, <>, in, not in]
Coding Date
[=, <>, >, >=, <, <=, between,latest]
DRG CC Indicator
[=, <>, in, not in]
DRG (APR)
[=, <>, >, >=, <, <=, in, not in, between]
DRG (MS)
[=, <>, >, >=, <, <=, in, not in, between]
DRG ROM
[=, <>, >, >=, <, <=, in, not in, between]
Race
[=, <>, in, not in]
Gender
[=, <>, in, not in]
Flag Group - Any
[=, <>, in, not in]
Flag Cateogry - Any
[=, <>, in, not in]
Has Query
[=, <>, in, not in]
Is DRG Reimb
True/False
Physician Group - Attending
[=, <>, in, not in]
Physician - Procedure
[=, <>, in, not in]
Physician Group - Procedre
[=, <>, in, not in]
POA Indicator
[=, <>, in, not in]
Random Sample - Size
[=, <>, >, >=, <, <=, between,latest]
Review Date - Any
[=, <>, >, >=, <, <=, between]
Review Date - First
[=, <>, >, >=, <, <=, between]
Review Date - Latest
[=, <>, >, >=, <, <=, between]
Reviewer Group - Any
[=, <>, in, not in]
Reviewer - First
[=, <>, in, not in]
Reviewer Group - First
[=, <>, in, not in]
Reviewer - Latest
[=, <>, in, not in]
Reviewer Group - Latest
[=, <>, in, not in]
Secondary DRG Code (APR-DRG)
[=, <>, in, not in,between,like]
Secondary DRG Code (MS-DRG)
[=, <>, in, not in,between,like]
Secondary DRG  ROM
[=, <>, in, not in,between]
Updated By
[=,in,like,<>]
Work Queue State
[=,in,like,<>]

## 15) Database schemas
There will be staging, claims, and ml schema.  the staging schema will hold all of the processing metrics, claims, file names, which rules are turned on for a facility.  ml schema will be used for predictive analytics based on the processed claims.  claims schema will hold all of the processed claims, metrics, provider information, organization, region and facility information, coder information, reviewer information.  there's an organization hierarchy:  organization -> region -> facility.  organization is required, region is optional, facility is required.  there can be multiple regions.  a facility based on the facility_id can only be assigned to one region.  multiple facilities can be assigned to a region but the same facility cannot be assigned to different regions or organizations.  Ensure all views and tables are properly indexed in PostgreSQL.

## 16) Windows GUI installer
There needs to be a Windows GUI installer that checks to see if PostgreSQL is installed and any other prerequisities.  It'll ask for the database hostname, database name, database user, database password.  If postgresql isn't installed then indicate that to the user and stop the installer.  If it is installed, load the databae and schema but do it silently.  There should be no other windows opening up.  Log all of the installer steps in case it errors so it can be troubleshoot.

## 17) 837p Companion Guide
All data elements from the 837p_compguide.pdf need to be imported and follow good data structure in #15 database schemas.  Performance is paramount.