mod tokenizer;
use tokenizer::Tokenizer;


fn main() {
    Tokenizer::from_file("./model/tokenizer.bin".as_ref());
}
