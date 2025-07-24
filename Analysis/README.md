# Analysis directory
Contains Analyses of relationships between mesoscale organisation of precipitation and the large-scale atmosphere.

If you would like to do an analyss create a new directory here with a descriptive name.


There are some ueful utilities in the tools/ directory.

- utils.py: includes functions to read the satellite and ERA5 data in a simple way.
- atm.py:   includes some thermodynamic functions.

To use these tools, add the following to your preamble:

```
# Using local utilities
import sys
sys.path.append("/path/to/repo/meso-org/Analysis/tools")

import util
import functions
```
