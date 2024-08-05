use env_logger;
use log::LevelFilter;
mod tokenizer;
use tokenizer::Tokenizer;


fn main() {
    env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .init();
    let mut tokenizer = Tokenizer::from_file("./model/tokenizer.bin");
    let tokens = vec![106, 1645, 108, 107, 108, 106, 2516, 108];
    println!("{}", tokenizer.decode(&tokens));

    let my_string = "hello! World? 你好 (♡˙︶˙♡)(ﾉ>ω<)ﾉヽ(∀ﾟ )人(ﾟ∀ﾟ)人( ﾟ∀)人(∀ﾟ )人(ﾟ∀ﾟ)人( ﾟ∀)ﾉ ";
    let tokens = tokenizer.encode(&my_string);
    let recover = tokenizer.decode(&tokens);
    println!("{} -> {:?} -> {:?}", my_string, tokens, recover);
}
