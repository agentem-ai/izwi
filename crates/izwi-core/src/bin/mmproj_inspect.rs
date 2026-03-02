use std::fs::File;
use std::io::BufReader;

use candle_core::quantized::gguf_file;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: mmproj_inspect <path>");
    let mut reader = BufReader::new(File::open(&path)?);
    let content = gguf_file::Content::read(&mut reader)?;

    println!("file: {}", path);
    println!("metadata entries: {}", content.metadata.len());
    println!("tensors: {}", content.tensor_infos.len());

    let mut keys = content
        .metadata
        .keys()
        .map(|k| k.to_string())
        .collect::<Vec<_>>();
    keys.sort();
    println!("\n== metadata keys ==");
    for k in keys {
        let v = content.metadata.get(&k).unwrap();
        println!("{} = {:?}", k, v);
    }

    let mut tensor_names = content
        .tensor_infos
        .keys()
        .map(|k| k.to_string())
        .collect::<Vec<_>>();
    tensor_names.sort();
    println!("\n== tensor names ==");
    for name in tensor_names {
        let info = content.tensor_infos.get(&name).unwrap();
        println!("{}: {:?} {:?}", name, info.shape, info.ggml_dtype);
    }

    Ok(())
}
