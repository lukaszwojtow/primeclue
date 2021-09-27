use primeclue::data::importer::{build_data_set, ClassRequest};
use primeclue::error::PrimeclueErr;
use primeclue::user::Settings;
use std::fs;
use std::path::PathBuf;

// This example will import a file. This is done mainly to avoid browser GUI with big files.
// Usage:
// cargo run --release --example import_big
fn main() -> Result<(), PrimeclueErr> {
    // Data to import
    let path = "/tmp/data.csv";
    let name = "import_big_name";
    // Get string content
    println!("Reading file");
    let content = String::from_utf8(fs::read(PathBuf::from(path))?)
        .map_err(|e| format!("Error converting file content: {:?}", e))?;
    let class_request = ClassRequest::simple_csv_request(name, content, true);
    println!("Building data set");
    // Uncomment this to load custom reward and penalty columns
    // class_request.custom_reward_penalty_columns = true;
    // let length = <number of columns here>;
    // class_request.penalty_column = length;
    // class_request.import_columns[class_request.penalty_column - 1] = false;
    // class_request.reward_column = length - 1;
    // class_request.import_columns[class_request.reward_column - 1] = false;
    // class_request.class_column = length - 2;
    // class_request.import_columns[class_request.class_column - 1] = false;
    let data_set = build_data_set(&class_request)?;
    let path = Settings::new()?.data_dir().join(name);
    println!("Saving to {:?}", path);
    data_set.save_to_disk(&path, |p| {
        if p % 100 == 0 {
            println!("Saved {} points", p);
        }
        Ok(())
    })?;
    println!("Saved {} points", data_set.len());
    println!("Import successful");
    Ok(())
}
