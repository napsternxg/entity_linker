from typing import Dict, Optional
from taisti_linker.commons import (EntityType, LabelWithIRI, get_entity_type,
                                   read_brat_all_annotation_files,
                                   read_ner_annotation_file,
                                   read_taisti_dataset_csv)
from taisti_linker.ontology_parser import OntologyParser, OntologyParserCSV
from taisti_linker.similarity_calculator import SimilarityCalculator, SimilarityType
from taisti_linker.text_processor import TextProcessor

import argparse
import csv
import pickle
import os
from tqdm.auto import tqdm

class EntityLinker:
    def __init__(
        self,
        ontology_path: str,
        annotated_examples_base_path: str,
        ner_output_path: str,
        taisti_csv_path: str,
        min_acceptable_similarity: float = 0.5,
        ignore_not_linkable: bool = False,
        similarity_measure: SimilarityType = SimilarityType.JACCARD
    ):
        self.ontology_path = ontology_path
        self.annotated_examples_base_path = annotated_examples_base_path
        self.ner_output_path = ner_output_path
        self.taisti_csv_path = taisti_csv_path
        self.min_acceptable_similarity = min_acceptable_similarity
        self.ignore_not_linkable = ignore_not_linkable
        self.similarity_measure = similarity_measure
        self.ontology_parser = OntologyParserCSV(ontology_path)
        self.text_processor = TextProcessor()
        self.similarity_calculator = SimilarityCalculator(
            similarity_measure, self.text_processor.normalize_text)
        self.cache = {}

        if len(ner_output_path) > 0:
            self.annotated_docs = read_ner_annotation_file(ner_output_path)
        elif len(annotated_examples_base_path) > 0:
            self.annotated_docs = \
                read_brat_all_annotation_files(annotated_examples_base_path)
        elif len(taisti_csv_path) > 0:
            self.annotated_docs = \
                read_taisti_dataset_csv(taisti_csv_path)

        self.normalized_label_mapping = \
            self.generate_label_mapping(self.text_processor)

    def link_all(self, output_path: str) -> None:
        """
            Iterate over internally stored annotated docs and link all spans marked by NER/BRAT to ontology entities.
            The result is then stored in a CSV file.

            Args:
                output_path (str): link to a CSV report file
        """
        print(f"INFO: Writing output to: {output_path}")
        f = open(output_path, "w")
        writer = csv.writer(f)

        for id, doc in enumerate(tqdm(self.annotated_docs, desc="Annotated Docs Link All")):
            if id % 500 == 0:
                print(f"Processing step: {id}")
            for annotation_id, annotation in enumerate(doc.annotations):
                # print(f"Processing step {id}/{annotation_id}")
                entity_text = annotation.text
                entity_type = get_entity_type(annotation.category)
                normalized_entity_text = \
                    self.text_processor.normalize_text(entity_text)

                linked_item: Optional[LabelWithIRI] = None
                if entity_type in self.normalized_label_mapping and normalized_entity_text in self.normalized_label_mapping[entity_type]:
                    linked_item = \
                        self.normalized_label_mapping[entity_type][normalized_entity_text]
                #    print(f"Direct match of {entity_text} to {linked_item}")
                else:
                    if normalized_entity_text not in self.cache:
                        linked_item = self.link(
                            normalized_entity_text, entity_type
                        )
                        self.cache[normalized_entity_text] = linked_item
                    else:
                        linked_item = self.cache[normalized_entity_text]

                annotation_data = [
                    annotation.file_id,
                    annotation.id,
                    annotation.category,
                    annotation.start,
                    annotation.end,
                    annotation.text,
                    annotation.source
                ]
                if linked_item:
                    writer.writerow(
                        annotation_data + [linked_item.iri, linked_item.label]
                    )
                elif not self.ignore_not_linkable:
                    writer.writerow(annotation_data + ["NONE", "NONE"])

    def link(
        self, text: str, entity_type: EntityType
    ) -> Optional[LabelWithIRI]:
        """
            Link a given piece of text accompanied by a category to some ontology Entity

            Args:
                text (str): text to link
                entity_type (EntityType): NER/BRAT entity type assigned to a given text
            Returns:
                Optional[LabelWithIRI]: linked entity or None if nothing is linked
        """
        best_item = None
        max_similarity = -1.0

        if entity_type not in self.normalized_label_mapping:
            return best_item

        category_label_mapping = self.normalized_label_mapping[entity_type]

        text_preprocessed = self.similarity_calculator.preprocess(text)

        for _, item in category_label_mapping.items():
            # preprocess and cache representation required by similarity calculator
            if not item.similarity_representation:
                item.similarity_representation = self.similarity_calculator.preprocess(item.normalized_label)

            # preprocess current text
            current_label_similarity = self.similarity_calculator.calculate(
                text_preprocessed, item.similarity_representation)

            if (
                current_label_similarity > self.min_acceptable_similarity
                and current_label_similarity >= max_similarity
            ):
                max_similarity = current_label_similarity
                best_item = item
                #print(
                #    f"Found new best: {item.normalized_label} with similarity {current_label_similarity}")
            if current_label_similarity == 1.0:
                break
        return best_item

    def generate_label_mapping(self, text_processor: TextProcessor) -> Dict[EntityType, Dict[str, LabelWithIRI]]:
        """
            From an ontology file, generate a map relating normalized labels of entities to their IRIs. Provide separate maps for each category.
            Because the map generation process is time consuming, caching is introduced -- if cache is present ('./foodon_cache.pkl') the map is not
            calculated. BEWARE: If normalization process changes, or ontology changes -- you have to remove the cache to recalculate the mapping.

            Args:
                text_processor (TextProcessor): text processor used to normalize ontology labels
            Returns:
                Dict[EntityType, Dict[str, LabelWithIRI]]: For each allowed entity type (e.g., )
        """
        cache_path = './foodon_cache.pkl'

        if os.path.exists(cache_path):
            print(f"Loading parsed ontology from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        else:
            print("Parsing ontology, it may take some time...")
            normalized_label_mapping = \
                self.ontology_parser.get_IRI_labels_data_per_category(
                    normalizer=text_processor
                )
            with open(cache_path, 'wb') as f:
                pickle.dump(normalized_label_mapping, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Writing parsed ontology to cache: {cache_path}")
            return normalized_label_mapping


def main(ontology_path: str, annotations_path: str, output_file_path: str,
         ner_output: str, taisti_csv_path: str, ignore_not_linkable: bool,
         similarity_measure: SimilarityType):
    """ Entry point """
    el = EntityLinker(ontology_path, annotations_path, ner_output, taisti_csv_path,
                      ignore_not_linkable=ignore_not_linkable,
                      similarity_measure=similarity_measure)
    el.link_all(output_file_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-op', '--ontology_path',
                        help='Path to ontology that we want to link to',
                        type=str,
                        default='../foodon.owl')
    parser.add_argument('-ap', '--annotations_path',
                        help='Path to BRAT annotations folder',
                        type=str,
                        default='')

    parser.add_argument('-ner', '--ner_output',
                        help='Use NER output stored in a given file to process',
                        type=str,
                        default='')
    parser.add_argument('-taisti', '--taisti_csv',
                        help='Use TAISTI CSV dataset sertialization',
                        type=str,
                        default='')
    parser.add_argument('-out', '--output_file_path',
                        help='Path to generated output file',
                        type=str,
                        default='./report.csv')
    parser.add_argument('-ignore', '--ignore_not_linkable',
                        help='Do not serialize entities that cannot be linked',
                        type=bool,
                        default=False)
    parser.add_argument('-s', '--similarity',
                        help='Similarity measure: J: Jaccard, E: Everygrams, W: Wordnet',
                        type=str,
                        default='J')

    args = parser.parse_args()
    main(args.ontology_path, args.annotations_path, args.output_file_path,
         args.ner_output, args.taisti_csv, args.ignore_not_linkable,
         SimilarityCalculator.similarity_id_to_type(args.similarity))
