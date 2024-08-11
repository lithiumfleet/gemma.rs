use env_logger;
use log::LevelFilter;
mod tokenizer;
mod nn;
use nn::Gemma2ForCausalLM;


fn main() {
    env_logger::builder()
        .filter_level(LevelFilter::Info)
        .init();
    let model_path = "./model/converted/model.bin";
    let tokenizer_path = "./model/converted/tokenizer.bin";
    let mut gemma2 = Gemma2ForCausalLM::new(model_path, tokenizer_path);

    let prompt = "hello here is Jane.";
    let max_seqlen = 512;
    let temperature = 0.5;
    let top_p = 0.9;
    let top_k = 10;
    let output = gemma2.generate(prompt, max_seqlen, temperature, top_p, top_k);

    println!("{}\n{}", prompt, output);
}
