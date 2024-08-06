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

impl Token {
    pub fn new_default() -> Token {
        Token {
            index: u32::MAX,
            piece: String::new(),
            score: -1e10
        }
    }
    
}


pub struct Tokenizer {
    pub n_words:u32,
    pub bos_id:u32,
    pub eos_id:u32,
    pub pad_id:u32,
    pub words:Vec<Token>,
    pub sorted_words:Vec<Token>
}

impl Tokenizer {

    pub fn from_file(path:&str) -> Self {

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
            words,
            sorted_words: vec![]
        }

    }

    pub fn decode_id(&self, input_id:u32) -> String {
        self.words[input_id as usize].piece.clone()
    }

    pub fn decode(&self, input_ids:&Vec<u32>) -> String {
        let mut res = vec![];
        for id in input_ids {
            let token:Token = self.words[*id as usize].clone();
            if token.piece.starts_with("<0x") && token.piece.ends_with('>') && token.piece.len() == 6 {
                if let Ok(bytes) = u8::from_str_radix(&token.piece[3..5], 16) {
                    res.push(char::from(bytes).to_string());
                }
            } else {
                res.push(token.piece.replace("▁", " "));
            }
        }
        res.join("")
    }

    pub fn encode(&mut self, input_str:&str) -> Vec<u32> {
        if self.sorted_words.is_empty() {
            self.sorted_words = self.words.clone();
            self.sorted_words.sort_by(|a, b| a.piece.cmp(&b.piece));
        }

        // this is u+226 and u+95: "▁" "_"
        let converted_input = input_str.to_string().replace(" ", "▁");
        let mut token_vec:Vec<Token> = vec![];
        for c in converted_input.chars() {
            let c_str = c.to_string();
            match self.sorted_words.binary_search_by(|token| token.piece.cmp(&c_str)) {
                Ok(index) => token_vec.push(self.sorted_words[index].clone()),
                Err(_) => {
                    for b in c_str.into_bytes().iter() {
                        token_vec.push(self.words[*b as usize + 3].clone());
                    }
                },
            }
        }

        loop {
            let mut best_token = Token::new_default();
            let mut replace_pos:usize = usize::MAX;
            for i in 0..token_vec.len()-1 {
                let merged_str:String = format!("{}{}", self.decode_id(token_vec[i].index), self.decode_id(token_vec[i+1].index));
                if let Ok(index) = self.sorted_words.binary_search_by(|t| t.piece.cmp(&merged_str)) {
                    let merged_token = &self.sorted_words[index];
                    debug!("{} -> {} (cur best{})", merged_token.piece, merged_token.score, best_token.score);
                    if merged_token.score > best_token.score {
                        best_token = merged_token.clone();
                        replace_pos = i;
                    }
                }
            }

            if best_token.index == u32::MAX {
                break;
            }

            token_vec[replace_pos] = best_token;
            token_vec.remove(replace_pos+1);
        }
        
        token_vec.iter().map(|t| t.index).collect()
    }
    
}