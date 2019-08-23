"""
Tools that are to be used by all ml classes.
"""

import os.path


def expect(condition, debug_data, debug_interactively=False):
    """Expect a condition to be true, if not, either a debugger is started or an exception
        is raised providing debug_data. This is controlled by the debug_interactively parameter."""
    # Pass check
    if condition:
        return

    # Handle failure
    if debug_interactively:
        print("Expectation failed, starting debugger")
        import pdb as dbg
        dbg.Pdb(skip=['daan.ml.tools.*']).set_trace()
    else:
        if callable(debug_data):
            debug_data = debug_data()
        raise ValueError("Qualitative check failed (%s)" % str(debug_data))


def path_join(path, name, default_extension=".jpg", uri=False):
    """Join path and file name while deducing the correct extension either from the name or from
        the path. If extension cannot be deduced, use default_extension. If uri parameter is True,
        make sure the returned path is a valid file uri."""
    if name[0] == "/":
        return name

    ext = default_extension
    if "*" in path:
        path, ext = path.rsplit("*", 1)

    if "/" not in ext:
        if ext and ext[-1] == "!":
            ext = ext[:-1]
            if ext:
                name = name.rsplit(".", 1)[0]
        elif "." in name.rsplit("/", 1)[-1] and name.rsplit(".", 1)[1]:
            ext = ""

    path = os.path.join(path, name+ext)
    if uri and path[0] == "/":
        path = "file://"+path
    return path


def pdb(params, data):
    """Call pdb debugger to debug params and data"""
    import pdb as dbg
    dbg.set_trace()
    return {"params_nkeys": len(params), "data_ncols": len(data)},
