# import needed utility functions into main namespace
from .mltools.utils import *

# import "base" (abstract) classifier and regressor classes
from .mltools.base import *

# import feature transforms etc into sub-namespace
from .mltools import transforms
from .mltools.transforms import rescale  # useful to have at top namespace

# import "plot" functions into main namespace
from .mltools.plot import *

# import classifiers into sub-namespaces
try:
    from .mltools import bayes
except ImportError:
    pass

try:
    from .mltools import knn
except ImportError:
    pass

try:
    from .mltools import linear
except ImportError:
    pass

try:
    from .mltools import linearC
except ImportError:
    pass

try:
    from .mltools import nnet
except ImportError:
    pass

try:
    from .mltools import dtree
except ImportError:
    pass

try:
    from .mltools import ensembles
except ImportError:
    pass

# import clustering unsupervised learning algos
try:
    from .mltools import cluster
except ImportError:
    pass
