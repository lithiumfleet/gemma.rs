from sentencepiece import SentencePieceProcessor
import struct
import os

DEFAULT_TOKENIZER_PATH = "./model/tokenizer.model"
DEFAULT_OUTPUT_PATH = "./model/tokenizer.bin"

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
        print(f"Bos: {self.sp_model.IdToPiece(self.sp_model.bos_id())}")
        print(f"Eos: {self.sp_model.IdToPiece(self.sp_model.eos_id())}")
        print(f"Unk: {self.sp_model.IdToPiece(self.sp_model.unk_id())}")
        print(f"Pad: {self.sp_model.IdToPiece(self.sp_model.pad_id())}")

        with open(output_path, "wb") as f:
            f.write("GRTK".encode("utf-8"))
            f.write(struct.pack("I", self.vocab_size))
            for token, score in zip(tokens, scores):
                f.write(struct.pack("I", len(token)))
                f.write(token)
                f.write(struct.pack("f", score))

        print("Finish convert!")


if __name__ == "__main__":
    print("This is tokenizer convertor script.")
    tokenizer = Tokenizer(DEFAULT_TOKENIZER_PATH)
    tokenizer.convert(DEFAULT_OUTPUT_PATH)
