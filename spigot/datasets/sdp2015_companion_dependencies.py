from typing import Dict, List, Tuple
import logging
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp_models.syntax import UniversalDependenciesDatasetReader
from allennlp_models.syntax.semantic_dependencies.semantic_dependencies_reader import lazy_parse


logger = logging.getLogger(__name__)


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
                pos_tags = [word["pos"] for word in annotated_sentence]
                heads = [word["head"] for word in annotated_sentence]
                tags = [word["deprel"] for word in annotated_sentence]
                yield self.text_to_instance(tokens, pos_tags, list(zip(tags, heads)))
