// use zstd_seekable::{parallel_compress, Seekable};

use flate2::read::GzDecoder;
use rayon::prelude::*;
use sonic_rs::JsonValueTrait;
use zstd_seekable::{CompressedFrame, SeekableCStream};
use std::collections::BTreeSet;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use zstd::stream::read::Decoder;

fn zstd_stream(path: &str) -> Result<Box<impl Read>, std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let decoder = Decoder::new(reader)?;
    Ok(Box::new(decoder))
}

fn gz_stream(path: &str) -> Result<Box<impl Read>, std::io::Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let decoder = GzDecoder::new(reader);
    Ok(Box::new(decoder))
}

fn list_files_glob(glob_str: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let paths = glob::glob(glob_str)?;
    let mut files = Vec::new();
    for path in paths {
        files.push(path.unwrap().to_str().unwrap().to_string());
    }
    Ok(files)
}


// fn gz_compress_to_chunks<T: zstd_seekable::Dst>(file_path: &str, chunk_size: usize) -> impl Iterator<Item=CompressedFrame<T>> {
//     let stream = BufReader::with_capacity(CHUNK_SIZE, gz_stream(file_path).unwrap());
//     let mut buf_stream_lines = BufReader::new(stream);

//     // let mut line_buf = String::new();
//     // let mut valid_counter = 0;

//     let mut buf = vec![0; CHUNK_SIZE];

//     let iter = std::iter::from_fn(|| {
//         let length = {
//             let buffer = buf_stream_lines.fill_buf().unwrap();
//             // do stuff with buffer here
//             buffer.len()
//         };
//         if length == 0 {
//             None
//         } else {
//             buf_stream_lines.consume(length);
//             let chunk = Chunk {
//                 p: buf.as_ptr(),
//                 len: buf.len() as size_t,
//             };
//             Some(zstd_seekable::compress_frame(chunk.p, chunk.len, level).unwrap())
//         }
//     });
//     iter
// }



const CHUNK_SIZE: usize = 64 * 1024 * 1024;
fn gz_compress_to_file(input_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut stream = gz_stream(input_path)?;
    
    let mut cstream = SeekableCStream::new(10, 1024*1024).unwrap();
    let mut chunk_buffer = vec![0; CHUNK_SIZE];
    let mut output_buffer = vec![0; CHUNK_SIZE];

    let output_file = File::create(output_path)?;
    let mut output_writer = BufWriter::new(output_file);


    let mut newline_indices: Vec<usize> = Vec::new();

    let mut offset_sofar = 0;
    while let Ok(sz) = stream.read(&mut chunk_buffer) {
        if sz == 0 {
            break;
        }
        let mut end_idx = 0;
        while end_idx < sz {
            let compressed_chunk = cstream.compress(&mut output_buffer, &chunk_buffer[end_idx..sz]).unwrap();
            // assert!(compressed_chunk.1 == sz);
            output_writer.write_all(&output_buffer[..compressed_chunk.0])?;
            end_idx += compressed_chunk.1;
        }
        
        // go through chunk_buffer and find newline indices

        for i in 0..sz {
            if chunk_buffer[i] == b'\n' {
                newline_indices.push(offset_sofar + i);
            }
        }

        offset_sofar += sz;
        // output_writer.write_all(&output_buffer[..compressed_chunk.0])?;
    }

    dbg!("ending");
    let sz = cstream.end_stream(&mut output_buffer)?;
    output_writer.write_all(&output_buffer[..sz])?;

    let output_path_newline_indices = format!("{}_newline_indices.bin", output_path);
    let mut output_file_newline_indices = File::create(output_path_newline_indices)?;
    let mut output_writer_newline_indices = BufWriter::new(output_file_newline_indices);

    for idx in newline_indices {
        output_writer_newline_indices.write_all(&idx.to_le_bytes())?;
    }

    Ok(())
}

fn gz_compress_to_file_zstd(input_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // execute in bash
    let cmd = format!("rm -f {} && gunzip -c {} | zstd -0 - -o {}", output_path, input_path, output_path);
    let output = std::process::Command::new("bash").arg("-c").arg(cmd).output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Command failed: {}", String::from_utf8_lossy(&output.stderr))).into())
    }
}

//     while let Ok(_) = buf_stream_lines.read_exact(&mut buf) {
        
//     }
// }

fn read_nth_usize<T>(source: &mut T, n: usize) -> Result<usize, Box<dyn std::error::Error>> where T: Read + Seek {
    let mut buf = [0u8; 8];
    source.seek(SeekFrom::Start((n*8).try_into()?))?;
    source.read_exact(&mut buf)?;
    Ok(usize::from_le_bytes(buf))
}


fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let path = "../data/oalex/authors/updated_date=2024-08-23/part_000.gz";

    // let mut index_file = File::open("../data/oalex/seekable_compr_test.jsonl.zst_newline_indices.bin")?;
    // let nth_idx = read_nth_usize(&mut index_file, 150)?;
    // dbg!(nth_idx);

    // let mut seekable = zstd_seekable::Seekable::init_file("../data/oalex/seekable_compr_test.jsonl.zst")?;
    // let mut buf = [0u8; 100];
    // let sz = seekable.decompress(&mut buf, nth_idx as u64)?;
    // println!("decompressed size: {}", sz);
    // println!("decompressed: {}", std::str::from_utf8(&buf[..sz]).unwrap());


    // gz_compress_to_file(path, "../data/oalex/seekable_compr_test.jsonl.zst")?;
    // gz_compress_to_file_zstd(path, "../data/oalex/seekable_compr_test_base.jsonl.zst")?;


    // let gz_stream = gz_stream(path)?;
    // // print position of first newline
    // let mut buf = [0u8; 99999];
    // let mut reader = BufReader::new(gz_stream);
    // let newline_pos = if let Ok(bytes_read) = reader.read(&mut buf) {
    //     if let Some(newline_pos) = buf[..bytes_read].iter().position(|&x| x == b'\n') {
    //         println!("Position of first newline: {}", newline_pos);
    //         Ok::<usize, std::io::Error>(newline_pos)
    //     } else {
    //         Err(std::io::Error::new(std::io::ErrorKind::Other, "No newline found in the first 1024 bytes").into())
    //     }
    // } else {
    //     Err(std::io::Error::new(std::io::ErrorKind::Other, "Failed to read from the stream").into())
    // }.unwrap();




    // let mut seekable = zstd_seekable::Seekable::init_file("../data/oalex/seekable_compr_test.jsonl.zst")?;
    // let mut buf = [0u8; 100];
    // let sz = seekable.decompress(&mut buf, newline_pos as u64)?;
    // println!("decompressed size: {}", sz);
    // println!("decompressed: {}", std::str::from_utf8(&buf[..sz]).unwrap());



    // Ok(())

    // let files = list_files_glob("../data/oalex/authors/updated_date=*/part_*.gz")?;
    let files = vec!["../data/oalex/oa_authors_merged.zst"];
    // dbg!(&files);


    


    let res = files.par_iter().map(|file: &&str| -> BTreeSet<String> {
        let stream = zstd_stream(file).unwrap();
        // let stream = gz_stream(file).unwrap();

        let mut buf_stream_lines = BufReader::new(stream);

        let mut valid_counter = 0;
        let mut line_buf = String::new();

        let mut id_len_set = BTreeSet::new();
        while let Ok(_) = buf_stream_lines.read_line(&mut line_buf) {
            if line_buf.is_empty() {
                break;
            }

            let parsed_id = sonic_rs::get(&line_buf, ["id"]).map_err(|x| std::io::Error::new(std::io::ErrorKind::Other, format!("Error at line {}: {}\n\n{}", valid_counter, x, line_buf))).unwrap();
            // let parsed: sonic_rs::Value = sonic_rs::from_str(&line_buf).map_err(|x| std::io::Error::new(std::io::ErrorKind::Other, format!("Error at line {}: {}\n\n{}", valid_counter, x, line_buf)))?;
            let id = parsed_id.as_str().unwrap();
            // let id_len = parsed_id.as_str().unwrap().len() as i64;
            id_len_set.insert(id.to_string());
            valid_counter += 1;
            line_buf.clear();
            // println!("{:?}", parsed);
            // break;
        }

        println!("valid_counter: {}", valid_counter);
        // dbg!(&id_len_set);
        id_len_set
    }).reduce(|| BTreeSet::new(), |mut a, b| {
        a.extend(b);
        a
    });
    dbg!(&res.len());
    Ok(())
    



    // for line in bufStreamLines {
    //     println!("{:?}", line.unwrap());
    //     break;
    // }
    // debug print first 100 bytes
    // let mut buf = [0; 100];
    // stream.read_exact(&mut buf).unwrap();
    // println!("{:?}", buf);

    // let stream = zstd_stream("index.json.zst").unwrap();

    // let input = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut diam ante, sollicitudin a dolor et, volutpat elementum nulla. Etiam nec efficitur nibh, quis rutrum risus. Maecenas quis lorem malesuada, aliquet mi vel, viverra nunc. Donec et nulla sed velit sagittis varius. Suspendisse vestibulum, neque lobortis ornare vestibulum, orci turpis vulputate nisi, ut sodales neque purus eget magna. Nunc condimentum, diam eu consequat venenatis, est nisl semper lorem, et lobortis velit justo sed nulla. Nunc sit amet tempor nunc, vel posuere ipsum. Cras erat tortor, pulvinar ac pretium eu, auctor ac nibh. Duis iaculis porta magna, eu lobortis elit. Duis vitae massa eros. Nulla non magna accumsan, egestas quam sit amet, laoreet lectus.";
    // let mut output = Vec::new();
    // zstd_seekable::parallel_compress::<&mut Vec<u8>, [u8; 256]>(input, &mut output, 10, 4).unwrap();
    // let mut decomp = Vec::new();
    // let mut s = { Seekable::init_buf(&output).unwrap() };
    // for frame in 0..s.get_num_frames() {
    //     let size = s.get_frame_decompressed_size(frame);
    //     let n = decomp.len();
    //     decomp.extend(std::iter::repeat(0).take(size));
    //     s.decompress_frame(&mut decomp[n..], frame);
    // }
    // println!("{:?}", std::str::from_utf8(&decomp).unwrap());

    // assert_eq!(&input[..], &decomp[..])
}
