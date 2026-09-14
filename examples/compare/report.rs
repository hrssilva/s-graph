//! Formats benchmark results into a plain-text comparison table (and, on
//! request, a CSV rendering for downstream plotting) -- no table-formatting
//! crate needed for output this small.

pub struct Row {
    pub dataset: String,
    pub structure: String,
    pub metric: String,
    pub value: String,
}

pub struct Report {
    pub rows: Vec<Row>,
}

impl Report {
    pub fn new() -> Self {
        Report { rows: Vec::new() }
    }

    pub fn push(&mut self, dataset: &str, structure: &str, metric: &str, value: impl Into<String>) {
        self.rows.push(Row {
            dataset: dataset.to_string(),
            structure: structure.to_string(),
            metric: metric.to_string(),
            value: value.into(),
        });
    }

    pub fn print_table(&self) {
        let w_dataset = self.rows.iter().map(|r| r.dataset.len()).max().unwrap_or(7).max(7);
        let w_structure = self.rows.iter().map(|r| r.structure.len()).max().unwrap_or(9).max(9);
        let w_metric = self.rows.iter().map(|r| r.metric.len()).max().unwrap_or(6).max(6);
        let w_value = self.rows.iter().map(|r| r.value.len()).max().unwrap_or(5).max(5);

        println!("{:<w_dataset$}  {:<w_structure$}  {:<w_metric$}  {:<w_value$}", "dataset", "structure", "metric", "value");
        println!("{}", "-".repeat(w_dataset + w_structure + w_metric + w_value + 6));
        for r in &self.rows {
            println!("{:<w_dataset$}  {:<w_structure$}  {:<w_metric$}  {:<w_value$}", r.dataset, r.structure, r.metric, r.value);
        }
    }

    pub fn to_csv(&self) -> String {
        let mut out = String::from("dataset,structure,metric,value\n");
        for r in &self.rows {
            out.push_str(&format!("{},{},{},{}\n", r.dataset, r.structure, r.metric, r.value));
        }
        out
    }
}
