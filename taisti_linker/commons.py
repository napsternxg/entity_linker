from dataclasses import dataclass
from enum import Enum
from typing import Any, List
import json
import os
import pandas as pd
import re

def pd_dfs(df, parent):
  visited = set()
  def dfs(root):
    df_children = df[df["?parent"] == root]
    if df_children.shape[0] == 0:
      return []
    visited.add(root)
    yield root
    children = df_children["?class"].dropna().values
    for child in children:
      if child in visited:
        continue
      yield from dfs(child)
  return dfs(parent)

class EntityType(Enum):
    """ Common entity types supported by NER and BRAT annotations """
    FOOD = 1
    UNIT = 2
    QUANTITY = 3
    PROCESS = 4
    COLOR = 5
    PHYSICAL_QUALITY = 6
    DIET = 7
    PART = 8
    PURPOSE = 9
    TASTE = 10


class AnnotationSource(Enum):
    """ The source of annotations. Either coming from linguists (BRAT) or NER """
    BRAT = 1
    NER = 2
    TAISTI_CSV  = 3


@dataclass
class Annotation:
    """ Common annotation format, shared between BRAT and NER (for NER file_id is artificial id). """
    id: str
    file_id: int
    start: int
    end: int
    category: str
    text: str
    source: AnnotationSource


@dataclass
class AnnotatedDoc:
    """ Textual document with all the BRAT/NER annotations """
    id: int
    path: str
    text: str
    annotations: List[Annotation]


@dataclass
class LabelWithIRI:
    """ Tuple of ontology entity label and its IRI. Also a normalized label is present """
    label: str
    iri: str
    normalized_label: str
    similarity_representation: Any = None


def get_entity_type(category: str) -> EntityType:
    """
        Map BRAT categories to common NER/BRAT categories defined in EntityType class

        Args:
            category (str): category from NER or BRAT
        Returns:
            EntityType: category as a shared EntityType object

    """
    category = category.lower()
    if 'food' in category or category in ['possible_substite', 'example', 'trade_name', 'excluded', 'exclusive']:
        return EntityType.FOOD
    elif category == 'unit':
        return EntityType.UNIT
    elif category == 'quantity':
        return EntityType.QUANTITY
    elif category == 'process':
        return EntityType.PROCESS
    elif category == 'color':
        return EntityType.COLOR
    elif category == 'physical_quality':
        return EntityType.PHYSICAL_QUALITY
    elif category == 'diet':
        return EntityType.DIET
    elif category == 'part':
        return EntityType.PART
    elif category == 'purpose':
        return EntityType.PURPOSE
    elif category == 'taste':
        return EntityType.TASTE
    return EntityType.FOOD


def read_brat_all_annotation_files(folder_path: str) -> list[AnnotatedDoc]:
    annotations = []
    """
        Iterate over all BRAT annotations in a folder and parse them into a list of AnnotatedDocs

        Args:
            folder_path (str): Path to a folder with all annotations
        Returns:
            list[AnnotatedDoc]: List of parsed annotations
    """

    for filename in os.listdir(folder_path):
        f = os.path.join(folder_path, filename)
        if os.path.isfile(f) and f.endswith("txt"):
            ann_path = f"{f[:-4]}.ann"
            id = get_file_id(f)
            with open(f) as brat_file:
                text = brat_file.read()
            brat_annotations = read_brat_annotations_from_file(ann_path)
            annotated_doc = AnnotatedDoc(
                id=id, path=f, text=text, annotations=brat_annotations
            )
            annotations.append(annotated_doc)
    return annotations


def read_brat_annotations_from_file(file_path: str) -> List[Annotation]:
    """
        Read all BRAT annotations from a file and parse them into a list of Annotations

        Args:
            file_path (str): Path to a BRAT annotation file
        Returns:
            list[Annotation]: List of parsed annotations
    """
    annotations = []

    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("T"):
                # filter annotations other than tokens
                id, details, text = line.strip().split("\t")
                if ";" in details:
                    # skip discontinuous annotations
                    continue

                category, start, end = details.split()
                file_id = get_file_id(file_path)
                annotations.append(
                    Annotation(
                        id=id,
                        file_id=file_id,
                        start=int(start),
                        end=int(end),
                        category=category,
                        text=text,
                        source=AnnotationSource.BRAT
                    )
                )
    return annotations


def read_ner_annotation_file(file_path: str) -> list[AnnotatedDoc]:
    """
        Iterate over all NER annotations in a file and parse them into a list of AnnotatedDocs

        Args:
            file_path (str): Path to a file with NER output
        Returns:
            list[AnnotatedDoc]: List of parsed annotations
    """
    annotations = []
    with open(file_path) as f:
        documents = json.load(f)
        for i, doc in enumerate(documents):
            ner_annotations = []
            for j, entity in enumerate(doc['entities_list']):
                ner_annotations.append(Annotation(
                    id=str(j), file_id=i, start=entity['start'],
                    end=entity['end'], category=entity['label'],
                    text=entity['text'], source=AnnotationSource.NER))

            annotations.append(AnnotatedDoc(
                id=i, path=file_path, text=doc['text'], annotations=ner_annotations
            ))
    return annotations


def read_taisti_dataset_csv(file_path: str) -> list[AnnotatedDoc]:
    """
        Iterate over all NER annotations in a file and parse them into a list of AnnotatedDocs

        Args:
            file_path (str): Path to a file with NER output
        Returns:
            list[AnnotatedDoc]: List of parsed annotations
    """
    annotations = []
    idx = 0
    for df in pd.read_csv(file_path,
                 usecols=['ingredients_entities'], chunksize=1000):
        print(f"Processing {idx}")
        for _, row in df.iterrows():
            entities = json.loads(row['ingredients_entities'])

            ner_annotations = []
            for j, entity in enumerate(entities):
                if 'food' in entity['type'].lower():
                    ner_annotations.append(Annotation(
                        id=str(j), file_id=idx, start=entity['start'],
                        end=entity['end'], category=entity['type'],
                        text=entity['entity'], source=AnnotationSource.TAISTI_CSV))

            annotations.append(AnnotatedDoc(
                id=idx, path=file_path, text='', annotations=ner_annotations
            ))
            idx += 1
    return annotations


def get_file_id(path: str) -> int:
    """
        As BRAT annotations come with a fixed format {num}.txt or {num}.ann, we extract num value.

        Args:
            path (str): path to an annotation file
        Returns:
            int: file id
    """
    return int(re.split(r"[./]", path)[-2])
