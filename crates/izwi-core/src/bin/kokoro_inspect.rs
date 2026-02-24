use std::path::PathBuf;

use candle_core::pickle::{read_pth_tensor_info, PthTensors};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let model_dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/Users/lennex/Library/Application Support/izwi/models/Kokoro-82M"));

    let ckpt = model_dir.join("kokoro-v1_0.pth");
    println!("== checkpoint ==");
    println!("{}", ckpt.display());
    for key in [None, Some("state_dict")] {
        println!("-- read_pth_tensor_info key={key:?}");
        match read_pth_tensor_info(&ckpt, true, key) {
            Ok(info) => {
                println!("tensor count: {}", info.len());
                for ti in info.iter().take(80) {
                    println!(
                        "{} | {:?} | {:?}",
                        ti.name,
                        ti.dtype,
                        ti.layout.shape().dims()
                    );
                }
                if info.len() > 80 {
                    println!("... ({} more)", info.len() - 80);
                }
            }
            Err(err) => {
                println!("error: {err}");
            }
        }
    }

    println!();
    println!("== checkpoint submodules ==");
    for key in [
        "bert",
        "bert_encoder",
        "predictor",
        "text_encoder",
        "decoder",
    ] {
        println!("-- key={key:?}");
        match read_pth_tensor_info(&ckpt, false, Some(key)) {
            Ok(info) => {
                println!("tensor count: {}", info.len());
                let mut rows = info
                    .iter()
                    .map(|ti| {
                        format!(
                            "{} | {:?} | {:?}",
                            ti.name,
                            ti.dtype,
                            ti.layout.shape().dims()
                        )
                    })
                    .collect::<Vec<_>>();
                rows.sort();
                for row in rows.iter().take(80) {
                    println!("{row}");
                }
                if rows.len() > 80 {
                    println!("... ({} more)", rows.len() - 80);
                }
            }
            Err(err) => println!("error: {err}"),
        }
    }

    println!();
    let voice = model_dir.join("voices").join("af_heart.pt");
    println!("== voice ==");
    println!("{}", voice.display());
    for key in [None, Some("pack"), Some("state_dict")] {
        println!("-- read_pth_tensor_info key={key:?}");
        match read_pth_tensor_info(&voice, true, key) {
            Ok(info) => {
                println!("tensor count: {}", info.len());
                for ti in info.iter().take(40) {
                    println!(
                        "{} | {:?} | {:?}",
                        ti.name,
                        ti.dtype,
                        ti.layout.shape().dims()
                    );
                }
                if info.len() > 40 {
                    println!("... ({} more)", info.len() - 40);
                }
            }
            Err(err) => {
                println!("error: {err}");
            }
        }
    }

    println!();
    println!("== checkpoint lazy tensors ==");
    for key in [None, Some("state_dict")] {
        println!("-- PthTensors::new key={key:?}");
        match PthTensors::new(&ckpt, key) {
            Ok(pth) => {
                let mut names: Vec<_> = pth.tensor_infos().keys().cloned().collect();
                names.sort();
                println!("tensor count: {}", names.len());
                for name in names.iter().take(40) {
                    let info = &pth.tensor_infos()[name];
                    println!("{name} | {:?} | {:?}", info.dtype, info.layout.shape().dims());
                }
                if names.len() > 40 {
                    println!("... ({} more)", names.len() - 40);
                }
            }
            Err(err) => {
                println!("error: {err}");
            }
        }
    }

    println!();
    println!("== voice raw tensor ==");
    match candle_core::pickle::read_all(&voice) {
        Ok(all) => {
            println!("read_all tensor count: {}", all.len());
            for (name, t) in all {
                println!("{name} | {:?} | {:?}", t.dtype(), t.shape().dims());
            }
        }
        Err(err) => println!("read_all error: {err}"),
    }

    println!("== voice via PthTensors::new(None) names ==");
    match PthTensors::new(&voice, None) {
        Ok(pth) => {
            println!("tensor count: {}", pth.tensor_infos().len());
            for (name, info) in pth.tensor_infos() {
                println!("{name} | {:?} | {:?}", info.dtype, info.layout.shape().dims());
            }
        }
        Err(err) => println!("error: {err}"),
    }

    Ok(())
}
