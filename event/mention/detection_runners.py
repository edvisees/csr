import os


def guess_entity_form(upos, xpos):
    if not upos == '_':
        if upos == 'PRON':
            return 'pronominal'
        elif upos == 'PROPN':
            return 'named'
        else:
            return 'nominal'


class DetectionRunner:
    def __init__(self, config, token_vocab, tag_vocab, ontology):
        self.model_name = config.model_name
        self.model_dir = config.model_dir
        self.trainable = True
        self.model = None
        self.ontology = ontology
        self.init_model(config, token_vocab, tag_vocab)

    def init_model(self, config, token_vocab, tag_vocab):
        if self.model_name == 'cnn':
            import torch
            from event.mention.models.trainable_detectors import TextCNN
            self.model = TextCNN(config, tag_vocab.vocab_size(),
                                 token_vocab.vocab_size())
            # Load model here.
            if torch.cuda.is_available():
                self.model.cuda()
        elif self.model_name == 'frame_rule':
            from event.mention.models.rule_detectors import \
                FrameMappingDetector
            self.model = FrameMappingDetector(config, token_vocab)
            self.trainable = False
        elif self.model_name == 'marked_field':
            from event.mention.models.rule_detectors import \
                MarkedDetector
            self.model = MarkedDetector(config, token_vocab)
            self.trainable = False

    def predict(self, test_reader, csr, component_name=None):
        for data in test_reader.read_window():
            tokens, tags, features, l_word_meta, meta = data
            center = int(len(l_word_meta) / 2)
            token, span = l_word_meta[center]
            this_feature = features[center]

            event_type = self.model.predict(data)

            if not component_name:
                component_name = self.model_name

            if event_type:
                evm = csr.add_event_mention(
                    span, span, token, 'aida', this_feature[-1],
                    component=component_name
                )

                if not evm:
                    continue

                args = self.model.predict_args(center, event_type, data)

                for rel_type, predicted_arg in args.items():
                    if predicted_arg:
                        word_index, word_type = predicted_arg
                        arg_text, arg_span = l_word_meta[word_index]

                        upos = features[word_index][3]
                        xpos = features[word_index][4]
                        entity_form = guess_entity_form(upos, xpos)

                        csr.add_event_arg_by_span(
                            evm, arg_span, arg_span, arg_text, 'aida',
                            rel_type, component=self.model_name,
                            arg_entity_form=entity_form
                        )
