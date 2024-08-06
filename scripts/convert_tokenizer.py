import token
from sentencepiece import SentencePieceProcessor # type: ignore
import struct
import os

DEFAULT_TOKENIZER_PATH = "./model/tokenizer.model"
DEFAULT_OUTPUT_PATH = "./model/converted/tokenizer.bin"

class Tokenizer:
    def __init__(self, tokenizer_path):
        self.tokenizer_path = tokenizer_path
        self.sp_model = SentencePieceProcessor()   
        self.sp_model.LoadFromFile(tokenizer_path)
        self.vocab_size = self.sp_model.vocab_size()

    def convert(self, output_path):
        print(f"Reading from {self.tokenizer_path}")
        tokens, scores = list(), list()
        for i in range(self.vocab_size):
            token = self.sp_model.IdToPiece(i)
            score = self.sp_model.GetScore(i)
            token = token.encode('utf-8')
            tokens.append(token)
            scores.append(score)
        print(f"Vocab Size: {self.vocab_size}")
        bos_id = self.sp_model.bos_id()
        eos_id = self.sp_model.eos_id()
        pad_id = self.sp_model.pad_id()
        print(f"Bos {bos_id}: {self.sp_model.IdToPiece(bos_id)}")
        print(f"Eos {eos_id}: {self.sp_model.IdToPiece(eos_id)}")
        print(f"Pad {pad_id}: {self.sp_model.IdToPiece(pad_id)}")

        with open(output_path, "wb") as f:
            f.write("GRTK".encode("utf-8"))
            f.write(struct.pack("I", self.vocab_size))
            f.write(struct.pack("III", bos_id, eos_id, pad_id))
            for token, score in zip(tokens, scores):
                f.write(struct.pack("I", len(token)))
                f.write(token)
                f.write(struct.pack("f", score))

        print("Finish convert!")


if __name__ == "__main__":
    print("This is tokenizer convertor script.")
    tokenizer = Tokenizer(DEFAULT_TOKENIZER_PATH)
    # corpus = "Another benefit is that if let allows us to match non-parameterized enum variants. This is true even in cases where the enum doesn't implement or derive PartialEq. In such cases if Foo::Bar == a would fail to compile, because instances of the enum cannot be equated, however if let will continue to work."
    corpus = "hello world!"
    input_ids = tokenizer.sp_model.EncodeAsIds(corpus)
    recover = "|".join([tokenizer.sp_model.DecodeIds(input_id) for input_id in input_ids])
    print(f"{corpus} -> {input_ids} -> {recover}")

    tokenizer.convert(DEFAULT_OUTPUT_PATH)
