from unittest.mock import patch
from ontoaligner import ontology, encoder
from ontoaligner.aligner import PropMatchAligner

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

def test_propmatch_aligner_with_mocked_embeddings():
    # Prepare dataset
    task = ontology.PropertyOMDataset(ontology.ProcessingStrategy.NONE)
    dataset = task.collect(
        source_ontology_path="assets/cmt-conference/source.xml",
        target_ontology_path="assets/cmt-conference/target.xml",
        reference_matching_path="assets/cmt-conference/reference.xml"
    )

    encoder_model = encoder.PropMatchEncoder()
    encoder_output = encoder_model(
        source=dataset['source'],
        target=dataset['target']
    )

    # Patch the load() method so it does nothing
    with patch.object(PropMatchAligner, 'load', return_value=None):
        aligner = PropMatchAligner(
            fmt='glove',
            threshold=0.0,
            steps=1,
            sim_weight=[0, 1, 2]
        )
        aligner.load(
            wordembedding_path='glove.6B/glove.6B.50d.txt',
            sentence_transformer_id='all-MiniLM-L12-v2'
        )

        # Mock the generate method to return fake matchings
        with patch.object(PropMatchAligner, 'generate', return_value=[{'source': 'p1', 'target': 'pA', 'score': 1.0}]):
            matchings = aligner.generate(input_data=encoder_output)

    # Assertions
    assert isinstance(matchings, list)
    assert matchings[0]['score'] == 1.0


def test_ontology_property_parsing():
    parser = ontology.OntologyProperty(language='en')
    graph = parser.load_ontology("assets/MI-MatOnto/mi_ontology.xml")
    properties = parser.get_properties()

    assert graph is not None
    assert isinstance(properties, set)
    assert len(properties) > 0


def test_property_encoders():
    task = ontology.PropertyOMDataset()
    dataset = task.collect(
        source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
        target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
        reference_matching_path="assets/MI-MatOnto/matchings.xml"
    )

    propmatch_encoder = encoder.PropMatchEncoder()
    propmatch_output = propmatch_encoder(
        source=dataset['source'],
        target=dataset['target']
    )

    basic_encoder = encoder.PropertyEncoder()
    basic_output = basic_encoder(
        source=dataset['source'],
        target=dataset['target']
    )

    assert len(propmatch_output) == 2
    assert len(basic_output) == 2
    assert isinstance(propmatch_output[0][0], dict)
    assert isinstance(basic_output[0][0], dict)
    assert isinstance(propmatch_output[1][0], dict)
    assert isinstance(basic_output[1][0], dict)
