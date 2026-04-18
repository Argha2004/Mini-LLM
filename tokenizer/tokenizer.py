import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="data/raw_train.txt",
    model_prefix="tokenizer/mini",
    vocab_size=50000,
    model_type="bpe",
    character_coverage=0.9995,
    input_sentence_size=5000000,
    shuffle_input_sentence=True
)