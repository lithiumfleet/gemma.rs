# gemma.rs

## Intro

This project aimming to run gemma using rust, ~~which can provide high performance to infer.~~

I apologize for the suboptimal performance of this code. It doesn't fully leverage Rust's capabilities. 
If you're looking for a more efficient implementation of Gemme2 that runs well on a computer, please visit [lmrs](https://github.com/samuel-vitorino/lm.rs/). 
This code is well structured and is primarily intended as a reference and learning tool for the Rust equivalent of [gemma_pytorch](https://github.com/google/gemma_pytorch) now.

## Reference

- Reference implement

[pytorch implementation](https://github.com/google/gemma_pytorch)

[lmrs](https://github.com/samuel-vitorino/lm.rs/)

[chat tempelate](https://ai.google.dev/gemma/docs/formatting?hl=zh-cn)

[chat tempelate in cpp](https://github.com/google/gemma.cpp/blob/main/gemma/common.cc#L130)

## TODO

+ [x] tokenizer
+ [x] model