use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use pro_parser_edi::EdiParser;
use pro_rvu::{RvuLookup, GpciLookup, PaymentCalculator};
use rust_decimal::Decimal;
use std::str::FromStr;

/// Generate multiple EDI claims in a single transaction
fn generate_edi_file(num_claims: usize) -> String {
    let mut edi = String::new();

    // ISA/GS headers
    edi.push_str("ISA*00*          *00*          *ZZ*SUBMITTERID    *ZZ*RECEIVERID     *240115*1430*^*00501*000000001*0*P*:~");
    edi.push_str("GS*HC*SUBMITTER*RECEIVER*20240115*1430*000000001*X*005010X222A1~");

    for i in 0..num_claims {
        let claim_num = (i + 1) as u32;
        edi.push_str(&format!(
            "ST*837*{:04}*005010X222A1~\
            BHT*0019*00*{:09}*20240115*1430*CH~\
            NM1*41*2*SUBMITTER NAME*****46*SUBID~\
            PER*IC*CONTACT*TE*5555551234~\
            NM1*40*2*RECEIVER NAME*****46*RECID~\
            HL*{}**20*1~\
            PRV*BI*PXC*207Q00000X~\
            NM1*85*2*BILLING PROVIDER*****XX*1234567890~\
            N3*123 MAIN ST~\
            N4*ANYTOWN*CA*12345~\
            REF*EI*123456789~\
            HL*{}*{}*22*0~\
            SBR*P*18*******CI~\
            NM1*IL*1*DOE*JOHN****MI*MEMBER{:09}~\
            N3*456 ELM ST~\
            N4*ANYTOWN*CA*12345~\
            DMG*D8*19800101*M~\
            NM1*PR*2*INSURANCE COMPANY*****PI*12345~\
            CLM*PCN{:09}*250.00***11:B:1*Y*A*Y*Y~\
            DTP*431*D8*20240115~\
            DTP*454*D8*20240115~\
            HI*ABK:Z00.00~\
            NM1*82*1*RENDERING*PROVIDER****XX*9876543210~\
            LX*1~\
            SV1*HC:99213*250.00*UN*1***1~\
            DTP*472*D8*20240115~\
            SE*31*{:04}~",
            claim_num,
            claim_num,
            i * 2 + 1,
            i * 2 + 2,
            i * 2 + 1,
            claim_num,
            claim_num,
            claim_num
        ));
    }

    // GS/ISA trailers
    edi.push_str(&format!("GE*{}*000000001~", num_claims));
    edi.push_str("IEA*1*000000001~");

    edi
}

fn bench_rvu_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rvu_calculation");

    // Initialize lookups
    let rvu_lookup = RvuLookup::default();
    let gpci_lookup = GpciLookup::default();
    let calculator = PaymentCalculator::new(rvu_lookup, gpci_lookup);

    group.bench_function("calculate_payment", |b| {
        b.iter(|| {
            black_box(
                calculator.calculate(
                    black_box("99213"),
                    black_box(2024),
                    black_box("00"),
                    black_box("11"),
                    black_box(vec![]),
                    black_box(Decimal::from_str("1.0").unwrap())
                )
            )
        });
    });

    group.bench_function("calculate_with_modifiers", |b| {
        b.iter(|| {
            black_box(
                calculator.calculate(
                    black_box("99213"),
                    black_box(2024),
                    black_box("00"),
                    black_box("11"),
                    black_box(vec!["25".to_string(), "59".to_string()]),
                    black_box(Decimal::from_str("1.0").unwrap())
                )
            )
        });
    });

    group.finish();
}

fn bench_full_pipeline_parse_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_parse_only");
    group.sample_size(10);

    for size in [100, 1000, 10000].iter() {
        let edi_data = generate_edi_file(*size);

        group.bench_with_input(BenchmarkId::new("parse", size), &edi_data, |b, data| {
            b.iter(|| {
                let mut parser = EdiParser::new();
                black_box(parser.parse(black_box(data)).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_throughput_claims_per_second(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_claims_per_second");
    group.sample_size(10);

    // Test various batch sizes to measure claims/sec
    for size in [100, 500, 1000, 5000, 10000].iter() {
        let edi_data = generate_edi_file(*size);

        group.bench_with_input(
            BenchmarkId::new("parsing_throughput", size),
            &edi_data,
            |b, data| {
                b.iter(|| {
                    let mut parser = EdiParser::new();
                    black_box(parser.parse(black_box(data)).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_memory_pressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pressure");
    group.sample_size(10);

    // Test with a very large file to measure memory handling
    let edi_data = generate_edi_file(10000);

    group.bench_function("10k_claims_memory", |b| {
        b.iter(|| {
            let mut parser = EdiParser::new();
            let result = parser.parse(black_box(&edi_data)).unwrap();
            // Access the data to ensure it's not optimized away
            black_box(result.claims.len())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_rvu_calculation,
    bench_full_pipeline_parse_only,
    bench_throughput_claims_per_second,
    bench_memory_pressure
);
criterion_main!(benches);
