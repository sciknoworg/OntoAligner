from ontoaligner import ontology, encoder

parser = ontology.OntologyProperty(language='en')

graph = parser.load_ontology("assets/MI-MatOnto/mi_ontology.xml")
properties = parser.get_properties()
print(f"Found {len(properties)} properties")


task = ontology.PropertyOMDataset()
dataset = task.collect(source_ontology_path="assets/MI-MatOnto/mi_ontology.xml",
                       target_ontology_path="assets/MI-MatOnto/matonto_ontology.xml",
                       reference_matching_path="assets/MI-MatOnto/matchings.xml")

print(dataset['source'][:5])

encoder_model = encoder.PropMatchEncoder()
encoder_output = encoder_model(source=dataset['source'], target=dataset['target'])
print(encoder_output['source'][0])


encoder_model = encoder.PropertyEncoder()
encoder_output = encoder_model(source=dataset['source'], target=dataset['target'])
print(encoder_output['source'][0])
