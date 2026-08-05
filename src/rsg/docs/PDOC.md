# API documentation

All Python modules contain module docstrings and public classes/functions carry
docstrings suitable for pdoc. Generate local API documentation from the package
source after activating the ROS environment:

```bash
pdoc nodes nodes.support.preprocessor nodes.support.phase1 -o docs/api
```

The C++ fuser uses Doxygen-compatible file and class comments.
