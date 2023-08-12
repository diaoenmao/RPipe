import yaml

global cfg
if 'cfg' not in globals():
    with open('config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

delimiters = ['_', '-', '~', ';', '#']


def process_args(args):
    for k in cfg:
        cfg[k] = args[k]
    if 'control_name' in args and args['control_name'] is not None:
        cfg['control'] = make_control(cfg['control'], args['control_name'])
    if cfg['control'] is not None:
        cfg['control_name'] = make_control_name(cfg['control'])
    return


def make_control(control, control_name):
    """Reconstruct the control dictionary using the provided template and flattened string."""
    delimiters = ['_', '-', '~', ';', '#']

    def recursive_reconstruct(d, s, level=0):
        """Recursively reconstruct dictionary using the template structure."""
        reconstructed = {}

        for key in d:
            if not s:
                break  # If the flattened string segment is empty, stop the reconstruction for this level

            if isinstance(d[key], dict):  # If the value in the template is a dictionary
                # Calculate the number of top-level segments we should extract for this key
                num_segments = len(d[key])
                segments = s.split(delimiters[level], num_segments)
                segment_for_key = segments[0]
                sub_dict = recursive_reconstruct(d[key], segment_for_key, level + 1)
                if sub_dict:  # Only add the sub-dictionary if it has content
                    reconstructed[key] = sub_dict
            else:  # If the value in the template is not a dictionary
                segment, _, s = s.partition(delimiters[level] if s and level < len(delimiters) else "")
                reconstructed[key] = segment
        return reconstructed

    return recursive_reconstruct(control, control_name)


def make_control_name(control):
    """Generate a flattened string of dictionary values."""

    def flatten_values_to_string(d, level=0):
        """Recursively traverse the dictionary and join the values based on depth."""
        # Base case: if the dictionary is empty or not a dictionary, return empty string
        if not isinstance(d, dict) or not d:
            return ""

        # Determine the delimiter for this level
        if level < len(delimiters):
            sep = delimiters[level]
        else:
            raise ValueError('Not valid level')

        # Recursive case: flatten each value in the dictionary
        parts = []
        for value in d.values():
            if isinstance(value, dict):
                flattened_value = flatten_values_to_string(value, level + 1)
                parts.append(flattened_value)
            else:
                parts.append(str(value))

        # Join the flattened parts with the delimiter for this level
        return sep.join(parts)

    control_name = flatten_values_to_string(control)
    # Removing the trailing delimiter if present
    if control_name and control_name[-1] in delimiters:
        return control_name[:-1]
    return control_name
