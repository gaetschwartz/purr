use purr_onnx_convert::convert::{Options, convert};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let (Some(input), Some(output_dir), None) = (args.next(), args.next(), args.next()) else {
        return Err("usage: purr-onnx-convert <model.onnx> <output-dir>".into());
    };
    let input = PathBuf::from(input);
    let name = input
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("the model path has no file name")?
        .to_owned();
    let data_name = format!("{name}_data");
    let output_dir = PathBuf::from(output_dir);
    std::fs::create_dir_all(&output_dir)?;

    let mut data = BufWriter::new(File::create(output_dir.join(&data_name))?);
    let mut graph = BufWriter::new(File::create(output_dir.join(&name))?);
    let summary = convert(
        &mut BufReader::new(File::open(&input)?),
        &mut data,
        &mut graph,
        &Options {
            data_location: &data_name,
        },
    )?;
    data.flush()?;
    graph.flush()?;
    println!("{summary:?}");
    Ok(())
}
