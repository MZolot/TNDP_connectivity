import math

BUS_SPEED_KM_H = 25
HUMAN_SPEED_KM_H = 5

BUS_SPEED_M_MIN = (BUS_SPEED_KM_H * 1000) / 60
HUMAN_SPEED_M_MIN = (HUMAN_SPEED_KM_H * 1000) / 60

def _is_invalid(x):
    return x is None or (isinstance(x, float) and math.isnan(x))

def fill_length(G):
    fixed_edges = 0

    for u, v, key, data in G.edges(keys=True, data=True):
        length_meter = data.get('length_meter')
        length = data.get('length')

        if _is_invalid(length_meter) or _is_invalid(length):
            if 'geometry' in data:
                seg_length = data['geometry'].length
                data['length_meter'] = seg_length
                data['length'] = seg_length
                fixed_edges += 1
            else:
                print(f"Edge ({u}, {v}, {key}) has no length and no geometry")

    print(f"Updated {fixed_edges} edges with missing or NaN length.")
    
def normalize_length(G):
    fixed_edges = 0

    for _, _, _, data in G.edges(keys=True, data=True):
        length_attr = data.get('length_meter')

        if isinstance(length_attr, list):
            data['length_meter'] = sum(length_attr)
            fixed_edges += 1

    print(f"Fixed {fixed_edges} edges.")
    
def assign_edge_weights(G, divisor, weight_attr = 'weight'):
    for _, _, _, data in G.edges(keys=True, data=True):
        length = data.get('length_meter') or data.get('length')
        data[weight_attr] = length / divisor
        
def postprocess_graph(G, pedestrian=True):
    fill_length(G)
    normalize_length(G)
    if pedestrian:
        assign_edge_weights(G, HUMAN_SPEED_M_MIN)
    else:
        assign_edge_weights(G, BUS_SPEED_M_MIN)