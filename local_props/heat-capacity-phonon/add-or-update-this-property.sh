#!/bin/sh

# This script should be run from the directory containing the property files (e.g. property.edn) to be added

property_dir=$PWD

python -c \
"import kim_property;
import os;
assert os.access(kim_property.__path__[0],os.W_OK), 'kim_property must be installed in an editable location';
from packaging import version;
from kim_edn import load;
from pprint import PrettyPrinter;
properties=kim_property.get_properties();
property=load('$property_dir/property.edn');
property_id=property['property-id'];
properties[property_id]=property;
if version.parse(kim_property.__version__) < version.parse('2.6.0'):
   kim_property.pickle.pickle_kim_properties(properties);
else:
   kim_property.ednify.ednify_kim_properties(properties);
PrettyPrinter().pprint(kim_property.get_properties()[property_id]);
print('\n\nSuccessfully pickled or ednified properties! Scroll up to check the property you just added.');"
