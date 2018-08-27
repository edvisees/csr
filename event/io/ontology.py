import rdflib
from rdflib import Namespace
import re
import logging
from event.util import remove_punctuation
from collections import defaultdict


class MappingLoader:
    def __init__(self):
        self.mappings = {}

    def get_aida_arg_map(self):
        return self.mappings['aida_arg']

    def get_aida_event_map(self):
        return self.mappings['aida_event']

    def load_arg_aida_mapping(self, mapping_file):
        self.mappings['aida_arg'] = defaultdict(list)

        with open(mapping_file) as seedling_mappings:
            for line in seedling_mappings:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                event_type = parts[0]
                arg_type = parts[1]
                constraints = parts[2]
                mapped_frames = parts[3:]

                c_event_type = self.canonicalize_type(event_type)
                for frame_res in mapped_frames:
                    parts = frame_res.split(':')
                    if len(parts) > 1:
                        frame = parts[0]
                        res = set(parts[1:])
                    else:
                        frame = frame_res
                        res = None

                    self.mappings['aida_arg'][
                        (c_event_type, frame)].append((arg_type, res))

    def load_event_aida_mapping(self, mapping_file):
        # This mapping is from other ontology to seedling events, in
        # canonical format.
        self.mappings['aida_event'] = {}
        with open(mapping_file) as seedling_mappings:
            for line in seedling_mappings:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                event_type = parts[0]
                c_event_type = self.canonicalize_type(event_type)

                for frame in parts[1:]:
                    self.mappings['aida_event'][frame] = c_event_type

    def create_canonical_types(self, types):
        """
        Create a type map from a canonical version to the variations.
        :return:
        """
        type_map = {}
        for event_type in types:
            c_type = self.canonicalize_type(event_type)
            type_map[c_type] = event_type
        return type_map

    def canonicalize_type(self, event_type):
        return remove_punctuation(event_type).lower()


class OntologyLoader:
    def __init__(self, ontology_file):
        logging.info("Loading ontology from : {}".format(ontology_file))

        self.g = rdflib.Graph()
        self.g.load(ontology_file, format='ttl')

        self.ldc_ont = 'ldcOnt'
        self.aida_common = 'aidaDomainCommon'
        self.rdf = 'rdf'
        self.rdfs = 'rdfs'
        self.owl = 'owl'
        self.schema = 'schema'

        self.namespaces = {}
        for prefix, ns in self.g.namespaces():
            self.namespaces[prefix] = Namespace(ns)

        self.prefixes = {}

        self.event_onto = {}
        self.relation_onto = {}

        self.entity_types = set()
        self.filler_types = set()
        self.labels = {}
        self.__load()

    def event_onto_text(self):
        text_dict = {}
        for event_key, content in self.event_onto.items():
            _, _, event = self.shorten(event_key)
            args_text = {}

            for arg_type_ref, arg_content in content.get('args', {}).items():
                _, _, arg_type = self.shorten(arg_type_ref)
                args_text[arg_type] = {'restrictions': set()}
                for restrict_ref in arg_content.get('restrictions', {}):
                    _, _, restrict = self.shorten(restrict_ref)
                    args_text[arg_type]['restrictions'].add(restrict)

            text_dict[event] = {'args': args_text}

        return text_dict

    def get_arg_restricts(self):
        arg_restrictions = {}
        for event_key, content in self.event_onto.items():
            _, _, event = self.shorten(event_key)

            for arg_type_ref, arg_content in content.get('args', {}).items():
                _, _, arg_type = self.shorten(arg_type_ref)
                arg_restrictions[(event, arg_type)] = []
                for restrict_ref in arg_content.get('restrictions', {}):
                    _, _, restrict = self.shorten(restrict_ref)
                    arg_restrictions[(event, arg_type)].append(restrict)
        return arg_restrictions

    def __find_subclassof(self, target_class):
        return [a for (a, b, c) in self.g.triples(
            (None, self.namespaces[self.rdfs].subClassOf, target_class)
        )]

    def __find_values(self, subj):
        return [c for (a, b, c) in self.g.triples(
            (subj, self.namespaces[self.owl].allValuesFrom, None)
        )]

    def shorten(self, uri):
        for prefix, ns in self.namespaces.items():
            if uri.startswith(ns):
                return prefix, ns, re.sub('^' + ns, '', uri)
        return '', '', uri

    def __unpack_list(self, list_node):
        items = set()
        for _, _, res in self.g.triples(
                (list_node, self.namespaces[self.rdf].first, None)):
            items.add(res)

        for _, _, rest in self.g.triples(
                (list_node, self.namespaces[self.rdf].rest, None)):
            items.update(self.__unpack_list(rest))

        return items

    def __get_arg_range(self, arg_role):
        # Load event arguments restrictions.
        restrictions = set()
        for _, _, arg_range in self.g.triples(
                (arg_role, self.namespaces[self.schema].rangeIncludes, None)):
            restrictions.add(arg_range)
        return restrictions

    def __load(self):
        # Load entity types.
        for subj in self.__find_subclassof(
                self.namespaces[self.aida_common].EntityType):
            self.entity_types.add(subj)

        # Load filler types.
        for subj in self.__find_subclassof(
                self.namespaces[self.ldc_ont].FillerType):
            self.filler_types.add(subj)

        # Load event types.
        for evm_type in self.__find_subclassof(
                self.namespaces[self.aida_common].EventType):
            if evm_type not in self.event_onto:
                self.event_onto[evm_type] = {'args': {}}

        arg_events = {}
        # Load event arguments.
        for arg_type in self.__find_subclassof(
                self.namespaces[self.aida_common].EventArgumentType
        ):
            for _, _, evm_type in self.g.triples(
                    (arg_type, self.namespaces[self.rdfs].domain, None)):
                self.event_onto[evm_type]['args'][arg_type] = {}
                arg_events[arg_type] = evm_type

        for arg_type, evm_type in arg_events.items():
            for _, _, restrictions in self.g.triples((
                    arg_type, self.namespaces[self.schema].rangeIncludes, None
            )):
                self.event_onto[evm_type]['args'][arg_type][
                    'restrictions'] = set()
                restrictions = self.__get_arg_range(arg_type)
                self.event_onto[evm_type]['args'][arg_type][
                    'restrictions'].update(restrictions)

            for _, _, label in self.g.triples((
                    arg_type, self.namespaces[self.rdfs].label, None
            )):
                self.event_onto[evm_type]['args'][arg_type][
                    'label'] = label

        # Load relation types.
        for relation_type in self.__find_subclassof(
                self.namespaces[self.aida_common].RelationType):
            if relation_type not in self.relation_onto:
                self.relation_onto[relation_type] = {'args': {}}

        arg_relations = {}
        # Load relation arguments.
        for arg_type in self.__find_subclassof(
                self.namespaces[self.aida_common].RelationArgumentType
        ):
            for _, _, relation_type in self.g.triples(
                    (arg_type, self.namespaces[self.rdfs].domain, None)):
                self.relation_onto[relation_type]['args'][arg_type] = {}
                arg_relations[arg_type] = relation_type

        for arg_type, relation_type in arg_relations.items():
            for _, _, restrictions in self.g.triples((
                    arg_type, self.namespaces[self.schema].rangeIncludes, None
            )):
                self.relation_onto[relation_type]['args'][arg_type][
                    'restrictions'] = set()
                restrictions = self.__get_arg_range(arg_type)
                self.relation_onto[relation_type]['args'][arg_type][
                    'restrictions'].update(restrictions)

            for _, _, label in self.g.triples((
                    arg_type, self.namespaces[self.rdfs].label, None
            )):
                self.relation_onto[relation_type]['args'][arg_type][
                    'label'] = label

        # Load labels.
        for origin, _, label in self.g.triples((
                None, self.namespaces[self.rdfs].label, None
        )):

            if origin.startswith(self.namespaces[self.ldc_ont]):
                if not str(origin) == str(self.namespaces[self.ldc_ont]):
                    self.labels[origin] = label

    def as_text(self, output_path):
        with open(output_path, 'w') as out:
            out.write('#Event:\n')
            for full_type, event_info in self.event_onto.items():
                _, _, evm_type = self.shorten(full_type)
                restricts = {}

                for full_arg_name, arg_info in event_info['args'].items():

                    _, _, arg_name = self.shorten(full_arg_name)
                    restricts[arg_name] = []

                    for restrict in arg_info['restrictions']:
                        _, _, ent_name = self.shorten(restrict)
                        restricts[arg_name].append(ent_name)

                for restrict_arg, l_restrict_ent in restricts.items():
                    out.write(
                        '%s\t%s\t%s\t' % (
                            evm_type, restrict_arg, ' '.join(l_restrict_ent)
                        )
                    )
                    out.write('\n')
            out.write('\n')

            out.write('#Entity\n')
            for full_type in self.entity_types:
                _, _, entity_type = self.shorten(full_type)
                out.write(entity_type + '\n')
            out.write('\n')

            out.write('#Filler\n')
            for full_type in self.filler_types:
                _, _, filler_type = self.shorten(full_type)
                out.write(filler_type + '\n')
            out.write('\n')

            out.write('#Relation\n')
            for full_type, relation_info in self.relation_onto.items():
                _, _, relation_type = self.shorten(full_type)
                restricts = {}

                for full_arg_name, arg_info in relation_info['args'].items():
                    _, _, arg_name = self.shorten(full_arg_name)
                    restricts[arg_name] = []

                    for restrict in arg_info['restrictions']:
                        _, _, ent_name = self.shorten(restrict)
                        restricts[arg_name].append(ent_name)

                for restrict_arg, l_restrict_ent in restricts.items():
                    out.write(
                        '%s\t%s\t%s\t' % (
                            relation_type, restrict_arg,
                            ' '.join(l_restrict_ent)
                        )
                    )
                    out.write('\n')
            out.write('\n')

    def as_brat_conf(self, conf_path, visual_path=None):
        """
        Demonstrate how to convert ontology to a Brat config.
        :return:
        """
        from collections import defaultdict
        grouped_ent_types = defaultdict(list)
        for full_type in self.entity_types:
            prefix, ns, short = self.shorten(full_type)
            short = short.replace('.', '_')
            grouped_ent_types[prefix].append((short, full_type))

        grouped_filler_types = defaultdict(list)
        for full_type in self.filler_types:
            prefix, ns, short = self.shorten(full_type)
            short = short.replace('.', '_')
            grouped_filler_types[prefix].append((short, full_type))

        grouped_evm_types = defaultdict(list)
        for full_type in self.event_onto.keys():
            prefix, ns, short = self.shorten(full_type)
            short = short.replace('.', '_')
            grouped_evm_types[prefix].append((short, full_type))

        grouped_relation_types = defaultdict(list)
        for full_type in self.relation_onto.keys():
            prefix, ns, short = self.shorten(full_type)
            short = short.replace('.', '_')
            grouped_relation_types[prefix].append((short, full_type))

        with open(conf_path, 'w') as out:
            out.write('[entities]\n\n')

            for onto, types in grouped_ent_types.items():
                out.write('!{}\n'.format(onto + '_entity'))
                for t, full_type in sorted(types):
                    out.write('\t' + t + '\n')
                out.write('\n')

            for onto, types in grouped_filler_types.items():
                out.write('!{}\n'.format(onto + '_filler'))
                for t, full_type in sorted(types):
                    out.write('\t' + t + '\n')
                # Put a special other type here.
                out.write('\tOTHER\n')
                out.write('\n')

            out.write('[relations]\n\n')
            for onto, types in grouped_relation_types.items():
                out.write('!{}\n'.format(onto + '_relation'))
                for t, full_type in sorted(types):
                    sep = '\t'

                    out.write(t + '\t')

                    args = self.relation_onto[full_type]['args']
                    for arg, arg_content in args.items():
                        arg_prefix, arg_ns, arg_type = self.shorten(arg)
                        restricts = arg_content['restrictions']
                        plain_res = []

                        for r in restricts:
                            _, _, restrict_type = self.shorten(r)
                            plain_res.append(restrict_type.replace('.', '_'))

                        out.write(
                            '{}{}:{}'.format(sep, arg_type.replace('.', '_'),
                                             '|'.join(plain_res))
                        )
                        sep = ', '

                    if len(args) == 1:
                        # Fill unspecified role with general entity.
                        out.write(', Arg:<ENTITY>')

                    out.write('\n')

                out.write('\n')

            out.write(
                'ENT_COREF\tArg1:<ENTITY>, Arg2:<ENTITY>\n'
            )
            out.write(
                'EVM_COREF\tArg1:<EVENT>, Arg2:<EVENT>\n'
            )
            out.write(
                '<OVERLAP>\tArg1:<ENTITY>, Arg2:<ENTITY>, '
                '<OVL-TYPE>:<ANY>\n'
            )
            out.write('\n')

            out.write('[attributes]\n\n')

            out.write('[events]\n\n')
            out.write('#Definition of events.\n\n')

            for onto, types in grouped_evm_types.items():
                out.write('!{}\n'.format(onto + '_event'))
                for t, full_type in sorted(types):
                    out.write('\t' + t)
                    sep = '\t'

                    args = self.event_onto[full_type]['args']
                    for arg, arg_content in args.items():
                        arg_prefix, arg_ns, arg_type = self.shorten(arg)

                        restricts = arg_content['restrictions']
                        plain_res = []

                        for r in restricts:
                            _, _, restrict_type = self.shorten(r)
                            plain_res.append(restrict_type.replace('.', '_'))

                        out.write(
                            '{}{}:{}'.format(sep, arg_type.replace('.', '_'),
                                             '|'.join(plain_res))
                        )
                        sep = ', '


                    # Add a general arg.
                    out.write(', Arg:<ENTITY>')

                    out.write('\n')

            # Put a special other type here.
            out.write('\tOTHER_EVENT\n')

        if visual_path:
            with open(visual_path, 'w') as out:
                out.write('[labels]\n\n')
                for origin, label in self.labels.items():
                    prefix, ns, t = self.shorten(origin)
                    out.write((t + ' | ' + label).replace('.', '_'))
                    if len(t) < len(label):
                        out.write(' | ' + t.replace('.', '_'))
                    out.write('\n')

                out.write('\n[drawing]\n')
                out.write('SPAN_DEFAULT	fgColor:black, bgColor:lightgreen, '
                          'borderColor:darken\n')

                for t in self.entity_types:
                    out.write('{}	fgColor:black, bgColor:yellow, '
                              'borderColor:darken\n'.format(t.split('#')[1]))

                for t in self.filler_types:
                    out.write('{}	fgColor:black, bgColor:lightblue, '
                              'borderColor:darken\n'.format(t.split('#')[1]))

                for t in self.event_onto:
                    out.write('{}	fgColor:black, bgColor:cyan, '
                              'borderColor:darken\n'.format(
                        t.split('#')[1].replace('.', '_')))

    def __find_arg_restrictions(self):
        pass


if __name__ == '__main__':
    from event import util

    util.set_basic_log()

    # ontology_path = 'https://raw.githubusercontent.com/isi-vista' \
    #                 '/gaia-interchange/master/src/main/resources' \
    #                 '/edu/isi/gaia/seedling-ontology.ttl'

    ontology_path = "https://tac.nist.gov/tracks/SM-KBP/2018/" \
                    "ontologies/SeedlingOntology"

    loader = OntologyLoader(ontology_path)
    loader.as_brat_conf('annotation.conf', 'visual.conf')
    loader.as_text('ontology.txt')
