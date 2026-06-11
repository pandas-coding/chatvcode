use std::path::PathBuf;

use chatvcode_cli::chatvcode_core::index_path;
use chatvcode_cli::chatvcode_parser::parse_source;
use chatvcode_cli::format_index_result;
use clap::Parser;

#[derive(Parser)]
struct Args {
    #[arg(default_value = ".")]
    path: PathBuf,
}

fn main() {
    let args = Args::parse();

    match index_path(&args.path, &parse_source) {
        Ok(result) => {
            print!("{}", format_index_result(&result));
        }
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}
