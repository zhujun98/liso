"""
Author: Jun Zhu
"""
import re


def generate_input(template, mapping, output=None, dry_run=False):
    """Generate the input file from a template.

    Patterns in the template input file should be put between
    '<' and '>'.

    The function should not raise if there is any in 'mapping' which
    does not appear in template.

    :param template: string
        Template
    :param mapping: dictionary
        A pattern-value mapping for replacing the pattern with
        value in the template file.
    :param output: string
        Path of the output file.
    :param dry_run: bool
        For a "dry_run" (True), the output file will be be generated.
        This is used for consistency check. Default = False.

    :return: the found pattern set.
    """
    found = set()
    template = list(template)
    for i in range(len(template)):
        while True:
            line = template[i]

            # Comment line starting with '!'
            if re.match(r'^\s*!', line):
                break

            left = line.find('<')
            right = line.find('>')
            comment = line.find('!')

            # Cannot find '<' or '>'
            if left < 0 or right < 0:
                break

            # If '<' is on the right of '>'
            if left >= right:
                break

            # In line comment
            if left > comment >= 0:
                break

            ptn = line[left + 1:right]
            try:
                template[i] = line.replace('<' + ptn + '>', str(mapping[ptn]), 1)
            except KeyError:
                raise KeyError("No mapping for <{}> in the template file!".format(ptn))

            found.add(ptn)

    if dry_run is False:
        # Generate the files when all patterns are replaced
        with open(output, 'w') as fp:
            for line in template:
                fp.write(line)

    return found


def check_templates(templates, mapping):
    """Compare the mapping keys and the patterns in all input templates.

    :param templates: list
        A list of beamline input file templates.
    :param mapping: dictionary
        A pattern-value mapping for replacing the pattern with
        value in the template file.
    """
    found = set()
    for template in templates:
        found = found.union(generate_input(template, mapping, dry_run=True))

    if found != mapping.keys():
        raise ValueError("Variables %s not found in the templates!" %
                         (mapping.keys() - found))
