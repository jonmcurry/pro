use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use pro_parser_edi::EdiParser;
use pro_parser_csv::{CsvParser, PredefinedMappings};

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

/// Generate CSV test data
fn generate_csv_data(num_rows: usize) -> String {
    let mut csv = String::from("patient_control_number,subscriber_first_name,subscriber_last_name,subscriber_id,payer_name,payer_id,service_date_from,service_date_to,procedure_code,charge_amount,units\n");

    for i in 0..num_rows {
        csv.push_str(&format!(
            "PCN{:09},JOHN,DOE,MEMBER{:09},INSURANCE COMPANY,12345,2024-01-15,2024-01-15,99213,250.00,1\n",
            i + 1,
            i + 1
        ));
    }

    csv
}

fn bench_edi_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("edi_parser");

    for size in [1, 10, 100, 1000].iter() {
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

fn bench_edi_parser_single_claim(c: &mut Criterion) {
    let mut group = c.benchmark_group("edi_parser_single");

    let edi_data = generate_edi_file(1);

    group.bench_function("parse_one_claim", |b| {
        b.iter(|| {
            let mut parser = EdiParser::new();
            black_box(parser.parse(black_box(&edi_data)).unwrap())
        });
    });

    group.finish();
}

fn bench_csv_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("csv_parser");

    for size in [1, 10, 100, 1000].iter() {
        let csv_data = generate_csv_data(*size);

        group.bench_with_input(BenchmarkId::new("parse", size), &csv_data, |b, data| {
            b.iter(|| {
                let mapping = PredefinedMappings::generic();
                let mut parser = CsvParser::new(mapping);
                black_box(parser.parse_reader(black_box(data.as_bytes())).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_csv_parser_single_row(c: &mut Criterion) {
    let mut group = c.benchmark_group("csv_parser_single");

    let csv_data = generate_csv_data(1);

    group.bench_function("parse_one_row", |b| {
        b.iter(|| {
            let mapping = PredefinedMappings::generic();
            let mut parser = CsvParser::new(mapping);
            black_box(parser.parse_reader(black_box(csv_data.as_bytes())).unwrap())
        });
    });

    group.finish();
}

fn bench_throughput_target(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_validation");
    group.sample_size(10); // Fewer samples for large benchmarks

    // Test against the target: 10,000 claims in 15 seconds
    let edi_data = generate_edi_file(10000);

    group.bench_function("10k_claims_target", |b| {
        b.iter(|| {
            let mut parser = EdiParser::new();
            black_box(parser.parse(black_box(&edi_data)).unwrap())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_edi_parser,
    bench_edi_parser_single_claim,
    bench_csv_parser,
    bench_csv_parser_single_row,
    bench_throughput_target
);
criterion_main!(benches);
