import geopandas as gpd

from iduedu import get_4326_boundary, get_intermodal_graph, get_drive_graph, get_walk_graph


def get_bounary_from_file(filename, crs=4326):
    data = gpd.read_file(filename)
    # .convex_hull.buffer(0.001)
    polygon = data.to_crs(crs).union_all()
    return polygon


def get_boundary_from_osm(osm_id):
    polygon = get_4326_boundary(osm_id=osm_id)
    return polygon


def get_streets_graph(
        boundary_polygon,
        graph_type,
        keep_largest_subgraph=True,
        clip_by_territory=True):

    if graph_type == 'drive':
        graph = get_drive_graph(territory=boundary_polygon,
                                keep_largest_subgraph=keep_largest_subgraph,
                                clip_by_territory=clip_by_territory)
    elif graph_type == 'walk':
        graph = get_walk_graph(territory=boundary_polygon,
                               keep_largest_subgraph=keep_largest_subgraph,
                               clip_by_territory=clip_by_territory)
    else:
        graph = get_intermodal_graph(territory=boundary_polygon,
                                     keep_largest_subgraph=keep_largest_subgraph,
                                     clip_by_territory=clip_by_territory)

    return graph
