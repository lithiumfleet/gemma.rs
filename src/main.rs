use env_logger;
use log::LevelFilter;
mod tokenizer;
use tokenizer::Tokenizer;


fn main() {
    env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .init();
    let tokenizer = Tokenizer::from_file("./model/tokenizer.bin");
}
