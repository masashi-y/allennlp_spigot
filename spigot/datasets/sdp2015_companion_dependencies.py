from typing import Dict, List, Tuple
import logging
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp_models.syntax import UniversalDependenciesDatasetReader
from allennlp_models.syntax.semantic_dependencies.semantic_dependencies_reader import lazy_parse
from allennlp.data.tokenizers import Token


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


@DatasetReader.register("sdp2015_dependencies")
class SDP2015CompanionDependencies(UniversalDependenciesDatasetReader):
    @overrides
    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        logger.info("Reading semantic dependency parsing data from: %s", file_path)

        with open(file_path) as sdp_file:
            for annotated_sentence, directed_arc_indices, _ in lazy_parse(sdp_file.read()):
                # If there are no arc indices, skip this instance.
                if not directed_arc_indices:
                    continue
                tokens = [word["form"] for word in annotated_sentence]
                pos_tags = [normalize_postag(word["pos"]) for word in annotated_sentence]
                heads = [int(word["head"]) for word in annotated_sentence]
                tags = [normalize_deprel(word["deprel"]) for word in annotated_sentence]
                yield self.text_to_instance(tokens, pos_tags, list(zip(tags, heads)))

    @overrides
    def text_to_instance(
        self,  # type: ignore
        words: List[str],
        upos_tags: List[str],
        dependencies: List[Tuple[str, int]] = None,
    ) -> Instance:

        """
        # Parameters

        words : `List[str]`, required.
            The words in the sentence to be encoded.
        upos_tags : `List[str]`, required.
            The universal dependencies POS tags for each word.
        dependencies : `List[Tuple[str, int]]`, optional (default = None)
            A list of  (head tag, head index) tuples. Indices are 1 indexed,
            meaning an index of 0 corresponds to that word being the root of
            the dependency tree.

        # Returns

        An instance containing words, upos tags, dependency head tags and head
        indices as fields.
        """
        fields: Dict[str, Field] = {}

        if self.tokenizer is not None:
            tokens = self.tokenizer.tokenize(" ".join(words))
        else:
            tokens = [Token(t) for t in words]

        text_field = TextField(tokens, self._token_indexers)
        fields["words"] = text_field
        fields["pos_tags"] = SequenceLabelField(upos_tags, text_field, label_namespace="pos")
        if dependencies is not None:
            # We don't want to expand the label namespace with an additional dummy token, so we'll
            # always give the 'ROOT_HEAD' token a label of 'root'.
            fields["head_tags"] = SequenceLabelField(
                [x[0] for x in dependencies], text_field, label_namespace="head_tags"
            )
            fields["head_indices"] = SequenceLabelField(
                [x[1] for x in dependencies], text_field, label_namespace="head_indices"
            )

        fields["metadata"] = MetadataField({"words": words, "pos": upos_tags})
        return Instance(fields)
