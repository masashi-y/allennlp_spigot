from typing import Dict, List, Tuple
import logging
from overrides import overrides

from allennlp_models.syntax import SemanticDependenciesDatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp_models.syntax.semantic_dependencies.semantic_dependencies_reader import lazy_parse

from allennlp.data.fields import AdjacencyField, MetadataField, SequenceLabelField
from allennlp.data.fields import Field, TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)


def normalize_postag(pos):
    if pos == '(':
        return '-LRB-'
    elif pos == ')':
        return '-RRB-'
    else:
        return pos


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
    """

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)

        logger.info("Reading semantic dependency parsing data from: %s", file_path)

        with open(file_path) as sdp_file:
            for annotated_sentence, directed_arc_indices, arc_tags in lazy_parse(sdp_file.read()):
                # If there are no arc indices, skip this instance.
                if not directed_arc_indices:
                    continue
                tokens = [word["form"] for word in annotated_sentence]
                pos_tags = [normalize_postag(word["pos"]) for word in annotated_sentence]
                heads = [int(word["head"]) for word in annotated_sentence]
                yield self.text_to_instance(
                    tokens, heads, pos_tags, directed_arc_indices, arc_tags)

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        heads: List[int],
        pos_tags: List[str] = None,
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
        if arc_indices is not None and arc_tags is not None:
            fields["arc_tags"] = AdjacencyField(arc_indices, token_field, arc_tags)

        return Instance(fields)
