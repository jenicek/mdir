def _dict_deep_overlay_item(original, key, item, list_replace):
    """Overlay a single item in original using given key. Param list_replace is passed
        to dict_deep_overlay()"""
    if isinstance(key, str) and key[-1] == "*":
        original[key[:-1]] = item
    elif isinstance(key, str) and key[-1] == "+":
        original[key[:-1]] += item
    elif key not in original:
        original[key] = item
    else:
        original[key] = dict_deep_overlay(original[key], item, list_replace=list_replace)



def dict_deep_overlay(*data, list_replace=False):
    """Overlay dictionaries deeply"""
    if len(data) == 1:
        return data[0]
    elif len(data) != 2:
        head = dict_deep_overlay(data[0], data[1], list_replace=list_replace)
        return dict_deep_overlay(head, *data[2:], list_replace=list_replace)

    original, overlay = data
    if isinstance(original, (list, tuple)) and isinstance(overlay, dict):
        for key, item in overlay.items():
            assert isinstance(key, int)
            original[key] = dict_deep_overlay(original[key], item)
    elif not isinstance(original, type(overlay)):
        return overlay
    elif isinstance(overlay, dict):
        for key, item in overlay.items():
            _dict_deep_overlay_item(original, key, item, list_replace)
    elif isinstance(overlay, list) and not list_replace:
        raise ValueError("Cannot implicitly merge two lists, use key* or key+ " + \
                         "when inheriting: (list1: %s, list2: %s)" % (str(original), str(overlay)))
    else:
        return overlay
    return original
