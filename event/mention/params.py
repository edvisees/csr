from traitlets.config import Configurable
from traitlets import (
    Unicode,
    Bool,
    Float,
    Integer,
    List
)


class DetectionParams(Configurable):
    model_name = Unicode(help='Name of the model').tag(config=True)

    resource_folder = Unicode(help='Resource directory').tag(config=True)
    model_dir = Unicode(help='Directory to save and load model').tag(
        config=True)

    train_files = List(help='List of training input.').tag(config=True)
    dev_files = List(help='List of development files.').tag(config=True)
    test_folder = Unicode(help='Directory saving the test files.').tag(
        config=True)

    word_embedding = Unicode(help='Embedding size.').tag(config=True)
    word_embedding_dim = Integer(help='Word embedding dimension.',
                                 default_value=300).tag(config=True)

    position_embedding_dim = Integer(help='Position embedding dimension.',
                                     default_value=10).tag(config=True)

    tag_list = Unicode(help='List of possible tags to classify.').tag(
        config=True)
    tag_embedding_dim = Integer(help='Tag embedding dimension',
                                default_value=50).tag(config=True)

    dropout = Float(help='Dropout rate.', default_value=0.5).tag(config=True)
    context_size = Integer(help='Context size to search argument.',
                           default_value=30).tag(config=True)
    window_sizes = List(help='Window sizes for mention detection.',
                        default_value=[2, 3, 4, 5]).tag(config=True)
    filter_num = Integer(help='Number of filters in CNN detection.',
                         default_value=100).tag(config=True)
    fix_embedding = Bool(help='Whehter to fix the word embeddings.',
                         default_value=False).tag(config=True)

    batch_size = Integer(help='Batch size to read data.').tag(config=True)

    input_format = Unicode(help='Format of input.').tag(config=True)
    no_punct = Bool(help='Ignore or no punctuations from input.',
                    default_value=False).tag(config=True)
    no_sentence = Bool(help='Whether no sentence split from input.',
                       default_value=False).tag(config=True)

    # Frame based detector.
    frame_lexicon = Unicode(help='Frame lexicon.').tag(config=True)
    event_list = Unicode(help='List of events.').tag(config=True)
    entity_list = Unicode(help='List of entities.').tag(config=True)
    relation_list = Unicode(help='Lexicon for relations.', ).tag(config=True)

    language = Unicode(help='Language of data.').tag(config=True)

