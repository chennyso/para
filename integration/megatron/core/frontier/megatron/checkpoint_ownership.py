def reindex_local_layer_keys(state, layers, offset, prefix):
    prefixes = tuple((f'{prefix}layers.{layer.layer_number - 1 - offset}.', f'{prefix}layers.{index}.')
                     for index, layer in enumerate(layers))
    result = {}
    for key, value in state.items():
        local = key
        for old, new in prefixes:
            if key.startswith(old):
                local = new + key.removeprefix(old)
                break
        if local in result:
            raise ValueError('region checkpoint ownership contains duplicate local keys')
        result[local] = value
    return result
