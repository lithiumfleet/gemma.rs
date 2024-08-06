use env_logger;
use log::LevelFilter;
mod tokenizer;
use tokenizer::Tokenizer;


fn main() {
    env_logger::builder()
        .filter_level(LevelFilter::Debug)
        .init();
    let mut tokenizer = Tokenizer::from_file("./model/tokenizer.bin");
    // let tokens = vec![106, 1645, 108, 107, 108, 106, 2516, 108];
    // println!("{}", tokenizer.decode(&tokens));

    // let my_string = "Another benefit is that if let allows us to match non-parameterized enum variants. This is true even in cases where the enum doesn't implement or derive PartialEq. In such cases if Foo::Bar == a would fail to compile, because instances of the enum cannot be equated, however if let will continue to work.";
    let my_string = "hello world!";
    let tokens = tokenizer.encode(&my_string);
    let recover = tokenizer.decode(&tokens);
    println!("{} -> {:?} -> {:?}", my_string, tokens, recover);
}
