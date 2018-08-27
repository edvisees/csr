from event.io.readers import (
    ConllUReader,
    Vocab
)
from event.io.csr import CSR
from event.mention.models.trainable_detectors import (
    TextCNN,
)
import logging
import os


class DetectionTrainer:
    def __init__(self, config, token_vocab, tag_vocab):
        self.model_name = config.model_name
        self.model_dir = config.model_dir
        self.trainable = True
        self.model = None
        self.init_model(config, token_vocab, tag_vocab)

        if not os.path.isdir(config.model_dir):
            os.makedirs(config.model_dir)

    def init_model(self, config, token_vocab, tag_vocab):
        if self.model_name == 'cnn':
            import torch
            self.model = TextCNN(config, tag_vocab.vocab_size(),
                                 token_vocab.vocab_size())
            if torch.cuda.is_available():
                self.model.cuda()

    def train(self, train_reader, dev_reader):
        if not self.trainable:
            return
        from event.mention import train_util
        train_util.train(self.model, train_reader, dev_reader, self.model_dir,
                         self.model_name)

    def eval(self, dev_reader):
        return 0

    def predict(self, test_reader, csr):
        for data in test_reader.read_window():
            tokens, tags, features, l_word_meta, meta = data

            # Found the center lemma's type and possible arguments in
            # the window.
            event_type, args = self.model.predict(data)

            center = int(len(l_word_meta) / 2)

            token, span = l_word_meta[center]

            if not event_type == self.model.unknown_type:
                extent_span = [span[0], span[1]]
                for role, (index, entity_type) in args.items():
                    a_token, a_span = l_word_meta[index]
                    if a_span[0] < extent_span[0]:
                        extent_span[0] = a_span[0]
                    if a_span[1] > extent_span[1]:
                        extent_span[1] = a_span[1]

                evm = csr.add_event_mention(span, span, token, 'aida',
                                            event_type, component='aida')

                if evm:
                    for role, (index, entity_type) in args.items():
                        a_token, a_span = l_word_meta[index]

                        csr.add_entity_mention(a_span, a_span, a_token, 'aida',
                                               entity_type=entity_type,
                                               component='implicit')

                        csr.add_event_arg_by_span(evm, a_span, a_span, a_token,
                                                  'aida', role,
                                                  component='Implicit')


def main(config):
    token_vocab = Vocab(config.experiment_folder, 'tokens',
                        embedding_path=config.word_embedding,
                        emb_dim=config.word_embedding_dim)

    tag_vocab = Vocab(config.experiment_folder, 'tag',
                      embedding_path=config.tag_list)

    train_reader = ConllUReader(config.train_files, config, token_vocab,
                                tag_vocab, config.language)
    dev_reader = ConllUReader(config.dev_files, config, token_vocab,
                              train_reader.tag_vocab, config.language)
    detector = DetectionTrainer(config, token_vocab, tag_vocab)
    detector.train(train_reader, dev_reader)

    #     def __init__(self, component_name, run_id, out_path):
    res_collector = CSR('Event_hector_frames', 1, config.output, 'belcat')

    test_reader = ConllUReader(config.test_files, config, token_vocab,
                               train_reader.tag_vocab)

    detector.predict(test_reader, res_collector)

    res_collector.write()


if __name__ == '__main__':
    from event import util

    parser = util.evm_args()

    arguments = parser.parse_args()

    util.set_basic_log()

    logging.info("Starting with the following config:")
    logging.info(arguments)

    main(arguments)
