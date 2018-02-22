"""
Author: Jun Zhu

"""
import os
import re
from .beamline import Beamline


def generate_input(beamline, mapping):
    """Generate the input file from a template.

    Patterns in the template input file should be put between
    '<' and '>'.

    :param beamline: BeamLine instance.
        Beamline.
    :param mapping: dictionary
        A pattern-value mapping for replacing the pattern with
        value in the template file.

    :return: the found pattern set.
    """
    if not isinstance(mapping, dict):
        raise TypeError("mapping must be a dictionary!")
    if not isinstance(beamline, Beamline):
        raise TypeError("beamline should be a Beamline object!")

    # delete the existing input file
    input_file = os.path.join(beamline.dirname, beamline.input_file)
    try:
        os.remove(input_file)
    except OSError:
        pass

    found = set()
    template = list(beamline.template)
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

    # Generate the files when all patterns are replaced
    with open(input_file, 'w') as fp:
        for line in template:
            fp.write(line)

    return found
