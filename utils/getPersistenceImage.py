import math
import gudhi.representations
import gudhi as gd
import numpy as np


def persistenceImage(point_cloud, rv_eplision, resolution, bandwidth_eplision, homology_group):
    point_cloud = point_cloud.astype(np.float32)
    # rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=rv_eplision)
    # rips_complex_tree = rips_complex.create_simplex_tree()
    # persistence_diagram = rips_complex_tree.persistence()

    acX = gd.AlphaComplex(points=point_cloud).create_simplex_tree()
    persistence_diagram = acX.persistence()

    # Calculate Bandwidth
    birth_death = []
    for i in persistence_diagram:
        # print(i)
        if i[0] == 1:
            birth_death.append(i[1][0])
            birth_death.append(i[1][1])

    birth_death_arr = np.array(birth_death)
    birth_death_min = birth_death_arr.min()
    birth_death_max = birth_death_arr.max()

    img_range_max = birth_death_max - birth_death_min

    sigma = math.sqrt((-(img_range_max / resolution) ** 2) / (2 * math.log(bandwidth_eplision)))

    persistence_image = gd.representations.PersistenceImage(bandwidth=sigma, weight=lambda x: x[1] ** 2,
                                                            im_range=[0, img_range_max, 0, img_range_max],
                                                            resolution=[100, 100])
    pi = persistence_image.fit_transform([acX.persistence_intervals_in_dimension(homology_group)])
    return pi
