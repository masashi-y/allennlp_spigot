
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
from spigot.differentiable_eisner import eisner
# from spigot.syntactically_informed_graph_parser import SyntacticallyInformedGraphParser
# from spigot.biaffine_parser import MyBiaffineDependencyParser


logger = logging.getLogger(__name__)


@Model.register("syntactic_then_semantic")
class SyntacticThenSemanticParser(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        syntactic_parser: Model, #  BiaffineDependencyParser,
        semantic_parser: Model,  #  SyntacticallyInformedGraphParser,
        share_text_field_embedder: bool = True,
        share_pos_tag_embedding: bool = True,
        decay_syntactic_loss: float = 0.5,
        freeze_syntactic_parser: bool = False,
        edge_prediction_threshold: float = 0.5,
        **kwargs
    ) -> None:
        super().__init__(vocab, **kwargs)
        assert not (freeze_syntactic_parser and (share_pos_tag_embedding or share_text_field_embedder)), \
            'when setting `freeze_syntactic_parser`, `share_pos_tag_embedding` and `share_text_field_embedder` must be disabled'
        self.syntactic_parser = syntactic_parser
        self.semantic_parser = semantic_parser
        self.syntactic_parser.requires_grad_(not freeze_syntactic_parser)
        if share_text_field_embedder:
            self.semantic_parser.text_field_embedder = \
                    self.syntactic_parser.text_field_embedder
            self.semantic_parser.requires_grad_()
        if share_pos_tag_embedding:
            self.semantic_parser._pos_tag_embedding = \
                    self.syntactic_parser._pos_tag_embedding
            self.semantic_parser._pos_tag_embedding.requires_grad_()
        self.decay_syntactic_loss = decay_syntactic_loss
        self.freeze_syntactic_parser = freeze_syntactic_parser
        self.edge_prediction_threshold = edge_prediction_threshold

    @overrides
    def forward(
        self,  # type: ignore
        words: TextFieldTensors,
        pos_tags: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
        arc_tags: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters
        words : TextFieldTensors, required
            The output of `TextField.as_array()`.
        pos_tags : torch.LongTensor, optional (default = None)
            The output of a `SequenceLabelField` containing POS tags.
        metadata : List[Dict[str, Any]], optional (default = None)
            A dictionary of metadata for each batch element which has keys:
                words : `List[str]`, required.
                    The original string tokens in the sentence.
        head_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape `(batch_size, sequence_length)`.
        head_indices: torch.LongTensor,
            The output of a `SequenceLabelField` containing syntactic head indices.
        arc_tags : torch.LongTensor, optional (default = None)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape `(batch_size, sequence_length, sequence_length)`.
        # Returns
        An output dictionary.
        """
        syntactic_outputs = self.syntactic_parser(
                words=words,
                pos_tags=pos_tags,
                metadata=metadata,
                head_tags=head_tags,
                head_indices=head_indices)
        attended_arcs = syntactic_outputs['attended_arcs']
        mask = get_text_field_mask(words)
        batch_size, _ = mask.size()
        mask_with_root_token = torch.cat(
                [mask.new_zeros((batch_size,1)), mask], dim=1)
        predicted_heads = eisner(attended_arcs, mask_with_root_token)
        semantic_outputs = self.semantic_parser(
                words=words,
                head_indices=predicted_heads[:, 1:],
                pos_tags=pos_tags,
                metadata=metadata,
                arc_tags=arc_tags)


        output_dict = {
            'heads': syntactic_outputs['heads'],
            'head_tags': syntactic_outputs['head_tags'],
            'arc_probs': semantic_outputs['arc_probs'],
            'arc_tag_probs': semantic_outputs['arc_tag_probs'],
            'mask': mask,
            'words': [meta['words'] for meta in metadata],
            'pos': [meta['pos'] for meta in metadata],
        }

        if head_indices is not None:
            output_dict['syntactic_arc_loss'] = syntactic_outputs['arc_loss']
            output_dict['syntactic_tag_loss'] = syntactic_outputs['tag_loss']

        if arc_tags is not None:
            output_dict['semantic_arc_loss'] = semantic_outputs['arc_loss']
            output_dict['semantic_tag_loss'] = semantic_outputs['tag_loss']

        if head_indices is not None and arc_tags is not None:
            if self.freeze_syntactic_parser:
                loss = semantic_outputs['loss']
            else:
                loss = semantic_outputs['loss'] + \
                        self.decay_syntactic_loss * syntactic_outputs['loss']
            output_dict['loss'] = loss

        return output_dict

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        arc_tag_probs = output_dict["arc_tag_probs"].cpu().detach().numpy()
        arc_probs = output_dict["arc_probs"].cpu().detach().numpy()
        mask = output_dict["mask"]
        lengths = get_lengths_from_binary_sequence_mask(mask)
        arcs = []
        arc_tags = []
        for instance_arc_probs, instance_arc_tag_probs, length in zip(
            arc_probs, arc_tag_probs, lengths
        ):

            arc_matrix = instance_arc_probs > self.edge_prediction_threshold
            edges = []
            edge_tags = []
            for i in range(length):
                for j in range(length):
                    if arc_matrix[i, j] == 1:
                        edges.append((i, j))
                        tag = instance_arc_tag_probs[i, j].argmax(-1)
                        edge_tags.append(self.vocab.get_token_from_index(tag, "labels"))
            arcs.append(edges)
            arc_tags.append(edge_tags)

        output_dict["arcs"] = arcs
        output_dict["arc_tags"] = arc_tags
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        precision, recall, f1_measure = \
                self.semantic_parser._unlabelled_f1.get_metric(reset)
        attachment_scores = self.syntactic_parser._attachment_scores.get_metrics(reset)
        metrics["precision"] = precision
        metrics["recall"] = recall
        metrics["f1"] = f1_measure
        metrics["UAS"] = attachment_scores["UAS"]
        return metrics
