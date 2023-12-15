from typing import List
from renard.pipeline import Pipeline
from renard.pipeline.tokenization import NLTKTokenizer
from renard.pipeline.ner import BertNamedEntityRecognizer, NEREntity



# transform a list of BIO tags into a list of NER entities
# example: tokens: ['Thibault', 'Roux', 'Arthur Amalvy', 'are', 'researchers'], bio_tags: ['B-PER', 'I-PER', 'B-PER', 'I-PER', 'O', 'O'] -> [NEREntity(tokens=['Thibault', 'Roux'], start_idx=0, end_idx=2, tag='PER'), NEREntity(tokens=['Arthur', 'Almavy'], start_idx=2, end_idx=4, tag='PER')]
def from_bio_to_ner_entities(
    tokens: List[str], bio_tags: List[str], resolve_inconsistencies: bool = True
) -> List[NEREntity]:
    """Extract NER entities from a list of BIO tags

    :param tokens: a list of tokens
    :param bio_tags: a list of BIO tags.  In particular, BIO tags
        should be in the CoNLL-2002 form (such as 'B-PER I-PER')

    :return: A list of ner entities, in apparition order
    """
    assert len(tokens) == len(bio_tags)

    entities = []
    current_tag: Optional[str] = None
    current_tag_start_idx: Optional[int] = None

    for i, tag in enumerate(bio_tags):
        if not current_tag is None and not tag.startswith("I-"):
            assert not current_tag_start_idx is None
            entities.append(
                NEREntity(
                    tokens[current_tag_start_idx:i],
                    current_tag_start_idx,
                    i,
                    current_tag,
                )
            )
            current_tag = None
            current_tag_start_idx = None

        if tag.startswith("B-"):
            current_tag = tag[2:]
            current_tag_start_idx = i

        elif tag.startswith("I-"):
            if current_tag is None and resolve_inconsistencies:
                current_tag = tag[2:]
                current_tag_start_idx = i
                continue

    if not current_tag is None:
        assert not current_tag_start_idx is None
        entities.append(
            NEREntity(
                tokens[current_tag_start_idx : len(tokens)],
                current_tag_start_idx,
                len(bio_tags),
                current_tag,
            )
        )

    return entities


# convert a list of NER entities into a list of BIO tags
def from_ner_entities_to_bio(
    tokens: List[str], entities: List[NEREntity]
) -> List[str]:
    """Convert a list of NER entities into a list of BIO tags

    :param tokens: a list of tokens
    :param entities: a list of NER entities

    :return: a list of BIO tags
    """
    bio_tags = ["O"] * len(tokens)

    for entity in entities:
        bio_tags[entity.start_idx] = "B-" + entity.tag
        for i in range(entity.start_idx + 1, entity.end_idx):
            bio_tags[i] = "I-" + entity.tag

    return bio_tags



def ner_annotate(pipeline: Pipeline, text: str) -> List[NEREntity]:
    out = pipeline(text)
    assert not out.entities is None
    return out.entities


pipeline = Pipeline(
    [NLTKTokenizer(), BertNamedEntityRecognizer()], progress_report=None
)

txt = "Thibault Roux Arthur Amalvy are researchers"
entities = ner_annotate(pipeline, txt)

bio_entities = from_ner_entities_to_bio(txt.split(" "), entities)
print(bio_entities)