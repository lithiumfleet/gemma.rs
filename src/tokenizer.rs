use std::fs;
use std::str;
use log::{debug, info, warn, error};


trait FromBytes {
    fn from_bytes(bytes: &[u8]) -> Self;
}

impl FromBytes for u32 {
    fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= 4);
        u32::from_le_bytes(bytes[..4].try_into().unwrap())
    }
}

impl FromBytes for f32 {
    fn from_bytes(bytes: &[u8]) -> Self {
        assert!(bytes.len() >= 4);
        f32::from_le_bytes(bytes[..4].try_into().unwrap())
    }
}

impl FromBytes for String {
    fn from_bytes(bytes: &[u8]) -> Self {
        String::from_utf8(bytes.to_vec()).unwrap()
    }
}

fn read_next_item<T: FromBytes>(file: &Vec<u8>, cursor: &mut usize, step: usize) -> T {
    let slice = &file[*cursor..*cursor + step];
    *cursor += step;
    T::from_bytes(slice)
}

fn read_next_u32(file: &Vec<u8>, cursor: &mut usize) -> u32 {
    read_next_item::<u32>(file, cursor, 4)
}

fn read_next_f32(file: &Vec<u8>, cursor: &mut usize) -> f32 {
    read_next_item::<f32>(file, cursor, 4)
}

fn read_next_str(file: &Vec<u8>, cursor: &mut usize, len: usize) -> String {
    read_next_item::<String>(file, cursor, len)
}




#[derive(Clone, Debug)]
pub struct Token {
    pub index:u32,
    pub piece:String,
    pub score:f32
}


pub struct Tokenizer {
    pub n_words:u32,
    pub bos_id:u32,
    pub eos_id:u32,
    pub pad_id:u32,
    pub words:Vec<Token>
}

impl Tokenizer {

    pub fn from_file(path:&str) -> Tokenizer {

        info!("Tokenizer load from {}", path);

        let file = fs::read(path).expect(&format!("Cannot read file from {}", path));
        let mut cursor:usize = 0;

        let head_str = read_next_str(&file, &mut cursor, 4);
        assert_eq!(head_str, "GRTK", "Tokenizer file is not in gemmars format.");

        let n_words = read_next_u32(&file, &mut cursor);
        debug!("n_words: {}", n_words);

        let bos_id = read_next_u32(&file, &mut cursor);
        debug!("bos_id: {}", bos_id);

        let eos_id = read_next_u32(&file, &mut cursor);
        debug!("eos_id: {}", eos_id);

        let pad_id = read_next_u32(&file, &mut cursor);
        debug!("pad_id: {}", pad_id);

        let mut words:Vec<Token> = vec![];
        let mut index = 0;
        while cursor < file.len() {
            let len = read_next_u32(&file, &mut cursor);
            let piece = read_next_str(&file, &mut cursor, len as usize);
            let score = read_next_f32(&file, &mut cursor);
            let new_token = Token {
                index,
                piece,
                score
            };
            words.push(new_token);
            index += 1;
        }

        Tokenizer {
            n_words,
            bos_id,
            eos_id,
            pad_id,
            words
        }

    }

    
}