from typing import Dict, Any, List, Tuple

from overrides import overrides
from io import StringIO

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer

# POS tags have a unified colour.
NODE_TYPE_TO_STYLE = {}

NODE_TYPE_TO_STYLE["root"] = ["color5", "strong"]
NODE_TYPE_TO_STYLE["dep"] = ["color5", "strong"]

# Arguments
NODE_TYPE_TO_STYLE["nsubj"] = ["color1"]
NODE_TYPE_TO_STYLE["nsubjpass"] = ["color1"]
NODE_TYPE_TO_STYLE["csubj"] = ["color1"]
NODE_TYPE_TO_STYLE["csubjpass"] = ["color1"]

# Complements
NODE_TYPE_TO_STYLE["pobj"] = ["color2"]
NODE_TYPE_TO_STYLE["dobj"] = ["color2"]
NODE_TYPE_TO_STYLE["iobj"] = ["color2"]
NODE_TYPE_TO_STYLE["mark"] = ["color2"]
NODE_TYPE_TO_STYLE["pcomp"] = ["color2"]
NODE_TYPE_TO_STYLE["xcomp"] = ["color2"]
NODE_TYPE_TO_STYLE["ccomp"] = ["color2"]
NODE_TYPE_TO_STYLE["acomp"] = ["color2"]

# Modifiers
NODE_TYPE_TO_STYLE["aux"] = ["color3"]
NODE_TYPE_TO_STYLE["cop"] = ["color3"]
NODE_TYPE_TO_STYLE["det"] = ["color3"]
NODE_TYPE_TO_STYLE["conj"] = ["color3"]
NODE_TYPE_TO_STYLE["cc"] = ["color3"]
NODE_TYPE_TO_STYLE["prep"] = ["color3"]
NODE_TYPE_TO_STYLE["number"] = ["color3"]
NODE_TYPE_TO_STYLE["possesive"] = ["color3"]
NODE_TYPE_TO_STYLE["poss"] = ["color3"]
NODE_TYPE_TO_STYLE["discourse"] = ["color3"]
NODE_TYPE_TO_STYLE["expletive"] = ["color3"]
NODE_TYPE_TO_STYLE["prt"] = ["color3"]
NODE_TYPE_TO_STYLE["advcl"] = ["color3"]

NODE_TYPE_TO_STYLE["mod"] = ["color4"]
NODE_TYPE_TO_STYLE["amod"] = ["color4"]
NODE_TYPE_TO_STYLE["tmod"] = ["color4"]
NODE_TYPE_TO_STYLE["quantmod"] = ["color4"]
NODE_TYPE_TO_STYLE["npadvmod"] = ["color4"]
NODE_TYPE_TO_STYLE["infmod"] = ["color4"]
NODE_TYPE_TO_STYLE["advmod"] = ["color4"]
NODE_TYPE_TO_STYLE["appos"] = ["color4"]
NODE_TYPE_TO_STYLE["nn"] = ["color4"]

NODE_TYPE_TO_STYLE["neg"] = ["color0"]
NODE_TYPE_TO_STYLE["punct"] = ["color0"]


LINK_TO_POSITION = {}
# Put subjects on the left
LINK_TO_POSITION["nsubj"] = "left"
LINK_TO_POSITION["nsubjpass"] = "left"
LINK_TO_POSITION["csubj"] = "left"
LINK_TO_POSITION["csubjpass"] = "left"

# Put arguments and some clauses on the right
LINK_TO_POSITION["pobj"] = "right"
LINK_TO_POSITION["dobj"] = "right"
LINK_TO_POSITION["iobj"] = "right"
LINK_TO_POSITION["pcomp"] = "right"
LINK_TO_POSITION["xcomp"] = "right"
LINK_TO_POSITION["ccomp"] = "right"
LINK_TO_POSITION["acomp"] = "right"


# exist_ok has to be true until we remove this from the core library
@Predictor.register("semantic_dependencies_predictor")
class SemanticDependenciesPredictor(Predictor):

    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)

    def dump_line(self, outputs: JsonDict) -> str:
        OFFSET = 8
        words = outputs['tokens']
        arcs = outputs['arcs']
        arc_tags = outputs['arc_tags']
        predicates = {i: {} for i in range(len(words))}
        for (i, j), tag in zip(arcs, arc_tags):
            predicates[j][i] = tag
        predicates = {
            predicate: deps for predicate, deps in predicates.items()
            if len(deps) > 0
        }
        lines = [
            [
                str(i + 1),
                word,
                word.lower(),
                'POS',
                '-',
                '+' if len(predicates[i]) > 0 else '-',
                '_'
            ] + ['_' for _ in range(len(predicates))]
            for i, word in enumerate(words)
        ]
        for head, deps in predicates.items():
            for dep, tag in deps.items():
                lines[dep][OFFSET + head] = tag
        return '\n'.join('\t'.join(line) for line in lines) + '\n'


    def predict(self, sentence: str) -> JsonDict:
        """
        Predict a semantic dependency parse for the given sentence.
        # Parameters

        sentence The sentence to parse.

        # Returns

        A dictionary representation of the dependency tree.
        """
        return self.predict_json({"sentence": sentence})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like `{"sentence": "..."}`.
        """
        raise NotImplementedError()

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)
