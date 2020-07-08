from typing import Dict, List, Tuple
import logging
from overrides import overrides

from allennlp_models.structured_prediction.dataset_readers.semantic_dependencies import SemanticDependenciesDatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp_models.structured_prediction.dataset_readers.semantic_dependencies import lazy_parse

from allennlp.data.fields import AdjacencyField, MetadataField, SequenceLabelField
from allennlp.data.fields import Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance

import random

logger = logging.getLogger(__name__)


def normalize_postag(pos):
    if pos == '(':
        return '-LRB-'
    elif pos == ')':
        return '-RRB-'
    else:
        return pos

def normalize_deprel(label):
    if label == 'vmod':
        return 'partmod'
    return label

def is_tree(indices):
    def rec(i):
        checked[i] = True
        for self, head in enumerate(indices, 1):
            if head == i:
                rec(self)
    checked = [False for _ in range(len(indices) + 1)]
    rec(0)
    return all(checked)


@DatasetReader.register("syntactic_then_semantic")
class SyntacticThenSemanticDependenciesdatasetReader(SemanticDependenciesDatasetReader):
    """
    Reads a file in the SemEval 2015 Task 18 (Broad-coverage Semantic Dependency Parsing) format.
    This is an extension of the original `SemanticDependenciesDatasetReader`, and outputs
    instance with additional field `head_indices` which is an SequenceLabelField object
    representing a syntactic dependency tree.

    Registered as a `DatasetReader` with name "syntactic_then_semantic".

    # Parameters

    token_indexers : `Dict[str, TokenIndexer]`, optional (default=`{"tokens": SingleIdTokenIndexer()}`)
        The token indexers to be applied to the words TextField.
    skip_when_no_arcs : `bool`, optional (default=`True`)
        If this is true, skip examples containing no semantic arcs.
    """
    def __init__(
        self,
        skip_when_no_arcs: bool = True,
        drop_random_syntactic_ratio: float = None,
        **kwargs) -> None:
        super().__init__(**kwargs)
        self.skip_when_no_arcs = skip_when_no_arcs
        assert (
            drop_random_syntactic_ratio is None
            or 0. <= drop_random_syntactic_ratio <= 1.
        ), '`drop_random_syntactic_ratio` must be within 0. and 1.'
        self.drop_random_syntactic_ratio = drop_random_syntactic_ratio

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading semantic dependency parsing data from: %s", file_path)

        drop_count, kept_count = 0, 0
        with open(file_path) as sdp_file:
            for annotated_sentence, directed_arc_indices, arc_tags in lazy_parse(sdp_file.read()):
                # If there are no arc indices, skip this instance.
                if self.skip_when_no_arcs and not directed_arc_indices:
                    continue
                tokens = [word["form"] for word in annotated_sentence]
                pos_tags = [normalize_postag(word["pos"]) for word in annotated_sentence]

                if (
                    self.drop_random_syntactic_ratio is not None
                    and random.random() <= self.drop_random_syntactic_ratio
                ):
                    head, head_tags = None, None
                    drop_count += 1
                else:
                    heads = [int(word["head"]) for word in annotated_sentence]
                    assert is_tree(heads), ', '.join(str(head) for head in heads)
                    head_tags = [normalize_deprel(word["deprel"])
                                 for word in annotated_sentence]
                    kept_count += 1

                yield self.text_to_instance(
                    words=tokens,
                    pos_tags=pos_tags,
                    heads=heads,
                    head_tags=head_tags,
                    arc_indices=directed_arc_indices,
                    arc_tags=arc_tags)
        logger.info(
            "%d/%d examples are with/without syntactic dependencies.",
            kept_count,
            drop_count)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        pos_tags: List[str],
        heads: List[int] = None,
        head_tags: List[str] = None,
        arc_indices: List[Tuple[int, int]] = None,
        arc_tags: List[str] = None,
    ) -> Instance:

        fields: Dict[str, Field] = {}
        token_field = TextField([Token(t) for t in words], self._token_indexers)
        fields["words"] = token_field
        fields["metadata"] = MetadataField({"words": words, "pos": pos_tags})
        if pos_tags is not None:
            fields["pos_tags"] = SequenceLabelField(pos_tags, token_field, label_namespace="pos")
        if heads is not None:
            fields["head_indices"] = SequenceLabelField(heads, token_field, label_namespace="head_indices")
        if head_tags is not None:
            fields["head_tags"] = SequenceLabelField(head_tags, token_field, label_namespace="head_tags")
        if arc_indices is not None and arc_tags is not None:
            fields["arc_tags"] = AdjacencyField(arc_indices, token_field, arc_tags)

        return Instance(fields)
