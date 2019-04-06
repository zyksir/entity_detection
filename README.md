# entity_detection

- for every sentence `who was the trump ocean club international hotel and tower named after`
- in `data`, we can find `dev.txt` and `dev.pt` and `word_vocab.pt`
  - in `dev.txt`  , we can find sentence like `what/O is/O a/O film/O directed/O by/O wiebke/I von/I carolsfeld/I ?/O`
  - in `dev.pt`, we can use `self.seqs, self.seq_lens, self.seq_labels = torch.load(infile)` , we can get three lists.
    - seq in seqs is a Tensor size(batch_size, seq_len)
    - seq_len in seq_lens is a Tensor size(batch_size)
    - seq_label in seq_labels is a Tensor size(batch_size, seq_len)
- 

