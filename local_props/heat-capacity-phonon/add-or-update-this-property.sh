#!/bin/sh

# This script should be run from the directory containing the property files (e.g. property.edn) to be added

property_dir=$PWD

cd /home/openkim/kim-property

python -c \
"from kim_property import get_properties;
from kim_property.pickle import pickle_kim_properties;
from kim_edn import load;
from pprint import PrettyPrinter;
properties=get_properties();
property=load('$property_dir/property.edn');
property_id=property['property-id']
properties[property_id]=property;
pickle_kim_properties(properties);
PrettyPrinter().pprint(get_properties()[property_id]);
print('\n\nSuccessfully pickled properties! Scroll up to check the property you just added.');"
