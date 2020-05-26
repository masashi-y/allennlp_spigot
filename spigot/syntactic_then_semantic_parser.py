
from typing import Dict, Tuple, Any, List
import logging
import copy

from overrides import overrides
import torch
from torch.nn.modules import Dropout
import numpy

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, Embedding, InputVariationalDropout
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.modules import FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, Activation
from allennlp.nn.util import min_value_of_dtype
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.training.metrics import F1Measure
import spigot
from spigot.syntactically_informed_graph_parser import SyntacticallyInformedGraphParser
from allennlp_models.syntax import BiaffineDependencyParser


logger = logging.getLogger(__name__)


@Model.register("syntactic_then_semantic")
class SyntacticThenSemanticParser(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        syntactic_parser: Model, #:  # BiaffineDependencyParser,
        semantic_parser: Model,  #: SyntacticallyInformedGraphParser,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.syntactic_parser = syntactic_parser
        self.semantic_parser = semantic_parser
        initializer(self)
        import IPython; IPython.embed(); exit(1)

    @overrides
    def forward(
        self,  # type: ignore
        tokens: TextFieldTensors,
        head_indices: torch.LongTensor,
        pos_tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        arc_tags: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters
        tokens : TextFieldTensors, required
            The output of `TextField.as_array()`.
        head_indices: torch.LongTensor,
            The output of a `SequenceLabelField` containing syntactic head indices.
        pos_tags : torch.LongTensor, optional (default = None)
            The output of a `SequenceLabelField` containing POS tags.
        metadata : List[Dict[str, Any]], optional (default = None)
            A dictionary of metadata for each batch element which has keys:
                tokens : `List[str]`, required.
                    The original string tokens in the sentence.
        arc_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape `(batch_size, sequence_length, sequence_length)`.
        # Returns
        An output dictionary.
        """
        pass
