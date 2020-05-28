
from allennlp_models.syntax.semantic_dependencies.semantic_dependencies_reader import lazy_parse


def _render_displacy_format(annotated_sentence, arc_indices, arc_tags):
    result = {
        'words': [
            {'text': token['form'], 'tag': token['pos']}
            for token in annotated_sentence
        ],
        'arcs': [
            {
                'start': min(child, head),
                'end': max(child, head),
                'label': tag,
                'dir': 'left' if min(child, head) == child else 'right'
            }
            for (child, head), tag in zip(arc_indices, arc_tags)
        ]
    }
    return result


def render_displacy_format(file_name):
    with open(file_name) as f:
        for tree in lazy_parse(f.read()):
            yield _render_displacy_format(*tree)
