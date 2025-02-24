from pathlib import Path
from typing import Iterable, Tuple, Callable, List, ClassVar, Dict
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab
from ir_explain.datasets.dataIterBase import PairwiseTrainDatasetBase, PointwiseTrainDatasetBase, ValTestDatasetBase
import ir_explain.datasets.glove


Input = Tuple[torch.LongTensor, torch.IntTensor]
Batch = Tuple[torch.LongTensor, torch.IntTensor, torch.LongTensor, torch.IntTensor]
PointwiseTrainInput = Tuple[Input, int]
PointwiseTrainBatch = Tuple[Batch, torch.FloatTensor]
PairwiseTrainInput = Tuple[Input, Input]
PairwiseTrainBatch = Tuple[Batch, Batch]
ValTestInput = Tuple[int, int, Input, int]
ValTestBatch = Tuple[torch.IntTensor, torch.IntTensor, Batch, torch.IntTensor]


def _get_single_input(query: str, doc: str, vocab: Vocab) -> Input:
    """Return a (query, document) pair for BERT, making sure the strings are not empty
    Args:
        query (str): The query
        doc (str): The document
    Returns:
        Input: Non-empty query and document
    """
    # empty queries or documents might cause problems later on
    if len(query.strip()) == 0:
        query = '(empty)'
    if len(doc.strip()) == 0:
        doc = '(empty)'

    query_index = vocab.sent2index(query)
    doc_index = vocab.sent2index(doc)

    return torch.LongTensor(query_index), torch.LongTensor(doc_index)


def _collate_simple(inputs: Iterable[Input], pad_id: int) -> Batch:
    """Tokenize and collate a number of single inputs, preprocessing and padding.
       Args:
           inputs (Iterable[Input]): The inputs
           tokenizer : pyserini default
       Returns:
           Batch: Input IDs, attention masks
       """
    queries, docs = zip(*inputs)
    query_len = [len(x) for x in queries]
    doc_len = [len(x) for x in docs]

    return pad_sequence(queries, batch_first=True, padding_value=pad_id), \
           torch.IntTensor(query_len), \
           pad_sequence(docs, batch_first=True, padding_value=pad_id), \
           torch.IntTensor(doc_len)


class PointwiseTrainDataset(PointwiseTrainDatasetBase):
    def __init__(self, data_file: Path, train_file: Path, vocab_dir: Path, max_length: int=510):
        super().__init__(data_file, train_file)


class PairwiseTrainDataset(PairwiseTrainDatasetBase):
    """Dataset for pairwise training.
    Args:
        data_file (Path): Data file containing queries and documents
        train_file (Path): Trainingset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, train_file: Path, vocab: Vocab, max_length: int=510):
        super().__init__(data_file, train_file)
        #self.vocab = glove.GloveTokenizer(data_file, vocab_dir, max_length)
        self.vocab = vocab
        self.pad_id = self.vocab.pad_id

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.
        Args:
            query (str): The query
            doc (str): The document
        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc, self.vocab)

    def collate_fn(self, inputs: Iterable[PairwiseTrainInput]) -> PairwiseTrainBatch:
        """Collate a number of pairwise inputs.
        Args:
            inputs (Iterable[PairwiseTrainInput]): The inputs
        Returns:
            PairwiseTrainBatch: A batch of pairwise inputs
        """
        pos_inputs, neg_inputs = zip(*inputs)
        return _collate_simple(pos_inputs, self.pad_id), _collate_simple(neg_inputs, self.pad_id)


class ValTestDataset(ValTestDatasetBase):
    """Dataset for glove validation/testing.
    Args:
        data_file (Path): Data file containing queries and documents
        val_test_file (Path): Validationset/testset file
        bert_type (str): Type for the tokenizer
    """
    def __init__(self, data_file: Path, val_test_file: Path, vocab: Vocab, max_length: int=510):
        super().__init__(data_file, val_test_file)
        self.vocab = vocab
        self.pad_id = self.vocab.pad_id

    def get_single_input(self, query: str, doc: str) -> Input:
        """Create a single model input from a query and a document.
        Args:
            query (str): The query
            doc (str): The document
        Returns:
            Input: The model input
        """
        return _get_single_input(query, doc, self.vocab)

    def collate_fn(self, val_test_inputs: Iterable[ValTestInput]) -> ValTestBatch:
        """Collate a number of validation/testing inputs.
        Args:
            val_test_inputs (Iterable[BertValInput]): The inputs
        Returns:
            ValTestBatch: A batch of validation inputs
        """
        q_ids, doc_ids, inputs, labels = zip(*val_test_inputs)
        return torch.IntTensor(q_ids), \
               torch.IntTensor(doc_ids), \
               _collate_simple(inputs, self.pad_id), \
               torch.IntTensor(labels)
