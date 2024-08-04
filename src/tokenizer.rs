use std::fs;
use std::str;

fn u32_from_bytes(bytes:&[u8]) -> u32 {
    assert!(bytes.len() >= 4);
    u32::from_le_bytes(bytes[..4].try_into().unwrap())
}

#[derive(Clone, Debug)]
pub struct Token {
    index:u32,
    piece:String,
    score:f32
}


pub struct Tokenizer {
    n_words:u32,
    bos_id:u32,
    eos_id:u32,
    pad_id:u32,
    words:Vec<Token>
}

impl Tokenizer {

    pub fn from_file(path:&str){

        println!("Parse from {}", path);

        let file = fs::read(path).expect(&format!("Cannot read file from {}", path));
        let mut cursor:usize = 0;

        let head_slice = &file[cursor..cursor+4];
        let head_str = String::from_utf8(head_slice.to_vec()).expect("File head is broken");
        assert_eq!(head_str, "GRTK", "Tokenizer file is not in gemmars format.");
        cursor += 4;

        let slice = &file[cursor..cursor+4];
        let n_words: u32 = u32_from_bytes(slice);
        cursor += 4;
        println!("n_words: {}", n_words);

        let slice = &file[cursor..cursor+4];
        let bos_id: u32 = u32_from_bytes(slice);
        cursor += 4;
        println!("bos_id: {}", bos_id);

        let slice = &file[cursor..cursor+4];
        let eos_id: u32 = u32_from_bytes(slice);
        cursor += 4;
        println!("eos_id: {}", eos_id);

        let slice = &file[cursor..cursor+4];
        let pad_id: u32 = u32_from_bytes(slice);
        cursor += 4;
        println!("pad_id: {}", pad_id);



    }

    
}