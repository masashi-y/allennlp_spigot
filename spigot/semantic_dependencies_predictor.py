from typing import Dict, Any, List, Tuple

from overrides import overrides
from io import StringIO

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.spacy_tokenizer import SpacyTokenizer


@Predictor.register("semantic_dependencies_predictor")
class SemanticDependenciesPredictor(Predictor):

    def __init__(
        self, model: Model, dataset_reader: DatasetReader, language: str = "en_core_web_sm"
    ) -> None:
        super().__init__(model, dataset_reader)
        self._tokenizer = SpacyTokenizer(language=language, pos_tags=True)

    def dump_line(self, outputs: JsonDict) -> str:
        OFFSET = 7
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
                '+' if i in predicates else '-',
                '_'
            ] + ['_' for _ in range(len(predicates))]
            for i, word in enumerate(words)
        ]
        for head, (_, deps) in enumerate(sorted(predicates.items())):
            for dep, tag in deps.items():
                lines[dep][OFFSET + head] = tag
        lines = '\n'.join('\t'.join(line) for line in lines)
        return f'#20000000\n{lines}\n\n'


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
        spacy_tokens = self._tokenizer.tokenize(json_dict["sentence"])
        sentence_text = [token.text for token in spacy_tokens]
        pos_tags = [token.tag_ for token in spacy_tokens]
        return self._dataset_reader.text_to_instance(sentence_text, pos_tags)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)
