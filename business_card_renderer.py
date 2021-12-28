# This code is downloaded from ...
# The code implements the ray casting idea from this post - http://fabiensanglard.net/rayTracing_back_of_business_card/
# using python + numba for speed.
# See https://medium.com/@sibearianpython/the-business-card-raytracer-in-python-8dcc868069f5 for background,
# and ... for the story of this code.
# See supplementary materials in the folder to understand better how 'tracer' works.
#
# The code is organized as follows:
# - imports
# - defining the rendered world - HERE you can render YOUR image! Read the blog posts above to know how
# - helper code for vector operations for python tuples
# - ray casting functions, starting from the deepest called function to the top-most function
#   ('create_frame' calls 'PMM_pixels_cycle' calls 'raycast' calls 'sample' calls 'tracer')
# - other helper functions
# - __main__

import os
import time
from typing import Tuple

import numpy as np
from math import sqrt, pow, ceil, pi, cos, sin, inf, floor
import random
from numba import njit
import cv2

# Describe the world
G_int = [16128, 49344, 65568, 394256, 525328, 1050632, 1050632, 2099208,
         2101252, 2101252, 2101252, 2105352, 1581064, 270344, 16392, 16400,
         16400, 16416, 32832, 33152, 33280, 64512, ]
# How many frames to render - the full circle would be 120 frames as defined inside 'create_frame' function
FRAMES = 10


# Here comes the code
Vector3D = Tuple[float, float, float]  # helper type for brevity


@njit(cache=True, fastmath=True)
def dot(x0, x1, x2, y0, y1, y2):
    return x0*y0 + x1*y1 + x2*y2


@njit(cache=True, fastmath=True)
def normalize(v0, v1, v2):
    coef = sqrt(dot(v0, v1, v2, v0, v1, v2))
    return v0 / coef, v1 / coef, v2 / coef


@njit(cache=True, fastmath=True)
def tracer(G,
           ray_origin: Vector3D, ray_direction: Vector3D,
           n: Vector3D,
           angle_sin: float, angle_cos: float) -> Tuple[int, float, Vector3D]:
    """This function traces the casted ray and checks its direction and whether it hits any of the spheres.

    Args:
        G: 2D numpy array representing where the spheres are located.
        ray_origin: coordinates of the ray origin, i.e. the camera or the previous hit by bouncing ray.
        ray_direction: vector of the direction of the casted ray.
        n: Default bouncing vector, or the bouncing vector of the previous tracing.
        angle_sin: precalculated sin(rotation angle) that defines the rotation for a given frame.
        angle_cos: precalculated cos(rotation angle) that defines the rotation for a given frame.

    Returns:
        m:
            0 if no hit was found but ray goes upward.
            1 if no hit was found but ray goes downward.
            2 if a hit was found (and also return minimal camera-sphere distance and bouncing vector).
        distance:
            distance to the closest object along the ray.
        n:
            bouncing vector from the hit object (if any).
    """
    # Floor is located as Z=0
    ray_direction = normalize(ray_direction[0], ray_direction[1], ray_direction[2])
    # First, let's calculate the return values as if there were no spheres
    if ray_origin[2] < -1e-10:
        # ray goes upward from the floor
        m = 0
        distance = 1e9
    elif ray_direction[2] < 0:
        # ray goes downward
        m = 1
        distance = - ray_origin[2] / ray_direction[2]  # distance to the floor
        n = (0, 0, 1)
    else:
        # ray goes upward not from the floor
        m = 0
        distance = inf
    # Now, let's iterate over the spheres and test if rays hits any of them
    # If yes - the return values will be changed accordingly
    rot_center_x = 2 + 0.65 * G_cols / 2  # rotation axis of the animation
    rot_center_y = 0
    for j in range(G_rows):
        for k in range(G_cols):
            if G[j][k]:
                # calculating offset for each frame based on the frame angle sin/cos
                r = 0.65 * (k - G_cols / 2)
                x = rot_center_x + r * angle_cos
                y = rot_center_y + r * angle_sin
                sphere_center = (x, y, 4 + 0.55 * j)
                ray_from_sphere_c_to_origin = (  # vector connecting the camera to the sphere center
                    ray_origin[0] - sphere_center[0], ray_origin[1] - sphere_center[1], ray_origin[2] - sphere_center[2]
                )
                # b = the distance between the camera and the closest point on the ray to the sphere center
                # i.e. b = the distance between the camera and the place where the casted ray
                # is hit by the perpendicular from the sphere center
                b = dot(
                    ray_from_sphere_c_to_origin[0], ray_from_sphere_c_to_origin[1], ray_from_sphere_c_to_origin[2],
                    ray_direction[0], ray_direction[1], ray_direction[2]
                )
                # c = square of the distance between the camera and the sphere center - radius squared
                # i.e., by right triangle equation,
                # c = the distance between the camera and the place where the tangential to the sphere
                # is hit by the perpendicular from the sphere center
                c2 = dot(
                    ray_from_sphere_c_to_origin[0], ray_from_sphere_c_to_origin[1], ray_from_sphere_c_to_origin[2],
                    ray_from_sphere_c_to_origin[0], ray_from_sphere_c_to_origin[1], ray_from_sphere_c_to_origin[2]
                ) - 0.4  # <--- sphere radius squared
                root2 = b * b - c2
                # from the definitions above, it can be shown that b^2 > c2 if and only if the ray intersects the sphere
                if root2 > 0:
                    camera_sphere_j_k_dist = - b - sqrt(root2)  # distance to the hit point
                    # if this sphere is hit closer than anything that was hit previously
                    # and the sphere is in front of the camera (will always be true)
                    if (camera_sphere_j_k_dist < distance) and (camera_sphere_j_k_dist > 1e-8):
                        m = 2
                        distance = camera_sphere_j_k_dist
                        # bouncing vector
                        n = (ray_from_sphere_c_to_origin[0] + ray_direction[0] * distance,
                             ray_from_sphere_c_to_origin[1] + ray_direction[1] * distance,
                             ray_from_sphere_c_to_origin[2] + ray_direction[2] * distance)
    if m == 2:
        # n contains the bouncing vector, it is normalized for the future calculations
        n = normalize(n[0], n[1], n[2])
    return m, distance, n


@njit(cache=True, fastmath=True)
def sample(G,
           ray_origin: Vector3D, ray_direction: Vector3D,
           angle_sin: float, angle_cos: float) -> Tuple[float, float, float]:
    """This function calls 'tracer' to trace the casted ray and returns the color of the place where the ray stops.

    Args:
        G: 2D numpy array representing where the spheres are located.
        ray_origin: coordinates of the ray origin, i.e. the camera or the previous hit by bouncing ray.
        ray_direction: vector of the direction of the casted ray.
        angle_sin: [not used, passed down to 'tracer']
        angle_cos: [not used, passed down to 'tracer']

    Returns:
        RGB-coded color of the hit pixel.
    """
    # Trace the ray
    m, distance, bouncing_ray = tracer(G, ray_origin, ray_direction, (0., 0., 0.), angle_sin, angle_cos)
    if m == 0:  # hit the sky
        coef = pow(1. - ray_direction[2], 4)  # adjustment coefficient to create gradient next to the horizon
        return 150 * coef, 180 * coef, 245 * coef  # RGB of the sky color is (150, 180, 245)
    # otherwise, a sphere maybe was hit
    # first - lets calculate how much light reaches the hit point
    intersection_coord = (ray_origin[0] + ray_direction[0] * distance,
                          ray_origin[1] + ray_direction[1] * distance,
                          ray_origin[2] + ray_direction[2] * distance)
    # shadow dithering
    light_source = (9. + 0.5 * random.random(),
                    9. + 0.5 * random.random(),
                    16.)
    direction_to_light = normalize(light_source[0] - intersection_coord[0],
                                   light_source[1] - intersection_coord[1],
                                   light_source[2] - intersection_coord[2])
    half_vector_coef = -2 * dot(bouncing_ray[0], bouncing_ray[1], bouncing_ray[2],
                                ray_direction[0], ray_direction[1], ray_direction[2])
    half_vector = normalize(
        ray_direction[0] + bouncing_ray[0] * half_vector_coef,
        ray_direction[1] + bouncing_ray[1] * half_vector_coef,
        ray_direction[2] + bouncing_ray[2] * half_vector_coef
    )
    color = 0.0
    lambertian_factor = dot(direction_to_light[0], direction_to_light[1], direction_to_light[2],
                            bouncing_ray[0], bouncing_ray[1], bouncing_ray[2])
    if lambertian_factor < 0:
        intensity = 0
    else:
        m2, _drop1, _drop2 = tracer(G, intersection_coord, direction_to_light, bouncing_ray, angle_sin, angle_cos)
        if m2 > 0:
            intensity = 0
        else:
            intensity = lambertian_factor * 0.85 + 0.35
            color = 245 * pow(dot(
                direction_to_light[0], direction_to_light[1], direction_to_light[2],
                half_vector[0], half_vector[1], half_vector[2]) * lambertian_factor, 25)
    # now 'intensity' and 'color' store the information about the brightness of the object
    # and what its color were if it was white respectively
    # now, we can apply this information to find out the color of the hit pixel
    if m == 1:
        # ray hits the floor directly - return the expected color of the floor at the hit coordinate
        # the following 'if' defines the pattern on the floor and its colors
        if int(ceil(intersection_coord[0]) // 2 + ceil(intersection_coord[1]) // 2) % 4 < 2:
            v = (135, 135, 50)
        else:
            v = (235, 235, 235)
        return intensity * v[0], intensity * v[1], intensity * v[2]
    else:  # m == 2
        # ray hits the sphere - bounce the ray (recursively sample) and attenuate the color
        s = sample(G, intersection_coord, half_vector, angle_sin, angle_cos)
        return color + s[0] * .7, color + s[1] * .7, color + s[2] * .7


@njit(cache=True, fastmath=True)
def raycast(G, x: int, y: int, angle_sin: float, angle_cos: float,
            camera_up_direction: Vector3D, right_direction: Vector3D,
            offset_from_the_eye_point: Vector3D, camera_focal_point: Vector3D):
    """This function renders one pixel (x, y) of the resulting image.
    It casts 'RAYS' rays through this pixel on the virtual screen (with slight noise each time)
    to sample the color of the pixel seen by the camera. It then aggregates the sampling results,
    and returns the RGB color of the pixel.

    Args:
        G: 2D numpy array representing where the spheres are located.
        x: X coordinate of the rendered pixel.
        y: Y coordinate of the rendered pixel.
        angle_sin: [not used, passed down to 'tracer']
        angle_cos: [not used, passed down to 'tracer']
        camera_up_direction: Camera orientation-defining vector.
        right_direction: Camera orientation-defining vector.
        offset_from_the_eye_point: Distance vector of the virtual screen from the camera.
        camera_focal_point: Camera location.

    Returns:
        RGB-coded color of the hit pixel.
    """
    # Defining ray casting parameters
    color = (30., 30., 30.)  # "black" color
    RAYS = 8  # number of rays to cast - use 24 for higher quality image (and 3x waiting time)
    color_adj = 1 / RAYS / 3
    sampled_colors = []
    for ray_i in range(RAYS):
        # random noise for each casted ray
        c = random.random() / 500.
        blur = (c * (camera_up_direction[0] + right_direction[0]),
                c * (camera_up_direction[1] + right_direction[1]),
                c * (camera_up_direction[2] + right_direction[2]))
        x_noised = random.random()/10. + x
        y_noised = random.random()/10. + y
        s = sample(
            G,
            (camera_focal_point[0] + blur[0],
             camera_focal_point[1] + blur[1],
             camera_focal_point[2] + blur[2],),  # camera location (noised)
            normalize(
                (camera_up_direction[0]/500. * x_noised + right_direction[0]/500. * y_noised
                 + offset_from_the_eye_point[0]) - blur[0],
                (camera_up_direction[1]/500. * x_noised + right_direction[1]/500. * y_noised
                 + offset_from_the_eye_point[1]) - blur[1],
                (camera_up_direction[2]/500. * x_noised + right_direction[2]/500. * y_noised
                 + offset_from_the_eye_point[2]) - blur[2],
            ),  # ray direction (noised)
            angle_sin, angle_cos,
        )
        sampled_colors.append(s)
    for s in sampled_colors:
        color = (s[0] * color_adj + color[0], s[1] * color_adj + color[1], s[2] * color_adj + color[2])
    return color


@njit(cache=True, fastmath=True)
def PMM_pixels_cycle(G, xdim, ydim, angle_sin, angle_cos,
                     camera_up_direction, right_direction, offset_from_the_eye_point, camera_focal_point):
    """This function calls 'raycast' for each pixel of the image and encodes its output in .ppm format.

    All the arguments are just passed to 'raycast'.
    Reverse order of the 'range' is defined by .ppm format.
    """
    arr = []
    for y in range(ydim, 0, -1):
        for x in range(xdim, 0, -1):
            color = raycast(
                G, x, y, angle_sin, angle_cos,
                camera_up_direction, right_direction, offset_from_the_eye_point, camera_focal_point
            )
            arr.append(floor(color[0]))
            arr.append(floor(color[1]))
            arr.append(floor(color[2]))
    return arr


@njit(cache=True, fastmath=True)
def create_frame(G, frame_n: int, xdim: int, ydim: int):
    """This function initializes the camera location and direction, and frame-specific sin/cos parameters
    (see more in the referred blog post).
    Then it calls the downstream function to render the world 'G' as it looks on frame 'frame_n'.

    Args:
        G: 2D numpy array representing where the spheres are located.
        frame_n: frame number.
        xdim: width of the rendered image.
        ydim: height of the rendered image.

    Returns:
        Bytes of the rendered image in .ppm format.
    """
    FULL_CIRCLE = 120  # how many frames we want full rotation to be
    # Defining the camera position and orientation
    camera_position = (17., 16., 8.)
    camera_direction = normalize(-6., -16., 0.)  # == (-0.351, -0.936, 0.0)
    camera_up_direction = np.cross((0., 0., 1.), camera_direction)
    camera_up_direction = \
        normalize(camera_up_direction[0], camera_up_direction[1], camera_up_direction[2])  # == (0.936, -0.351, 0.0)
    right_direction = np.cross(camera_direction, camera_up_direction)
    right_direction = \
        normalize(right_direction[0], right_direction[1], right_direction[2])  # == (0., 0., 1.)
    offset_from_the_eye_point = (
        (camera_up_direction[0] + right_direction[0]) * -0.5 + camera_direction[0],
        (camera_up_direction[1] + right_direction[1]) * -0.5 + camera_direction[1],
        (camera_up_direction[2] + right_direction[2]) * -0.5 + camera_direction[2]
    )  # == (-0.830, -0.756, -0.512)
    # Precalculating the rotation parameters for a given frame number - they will be used in 'tracer' function
    # to offset the spheres from their 'frame 0' location
    angle_radians = 2 * pi * frame_n / FULL_CIRCLE
    angle_sin = sin(angle_radians)
    angle_cos = cos(angle_radians)
    arr = PMM_pixels_cycle(G, xdim, ydim, angle_sin, angle_cos,
                           camera_up_direction=camera_up_direction,
                           right_direction=right_direction,
                           offset_from_the_eye_point=offset_from_the_eye_point,
                           camera_focal_point=camera_position)
    print(frame_n)
    return arr


def business_card_ray_tracing(G, prefix, frames):
    xdim, ydim = 512, 512  # dimensions of the frames
    for frame_n in range(frames):
        ppm_encoded_image = \
            bytes(f"P6 {xdim} {ydim} 255 ", encoding='utf-8') +\
            bytes(''.join(map(chr, create_frame(G, frame_n, xdim, ydim))), encoding='utf-8')
        with open(prefix + f'_{frame_n}.ppm', "wb") as ppm_fig:
            ppm_fig.write(ppm_encoded_image)
        cv2.imwrite(prefix + f"_{frame_n}.png", cv2.imread(prefix + f'_{frame_n}.ppm'))


def show_frames(fname, frames):
    images = [cv2.imread(fname + f'_{frame}.ppm') for frame in range(frames)]
    i = 0
    while True:
        cv2.imshow('window', images[i % len(images)])
        k = cv2.waitKey(33)  # Esc
        if k == 27:
            break
        i += 1


def prepare_the_world(world_int):
    world_str = [np.binary_repr(x) for x in world_int]
    width = max(len(s) for s in world_str)
    world_str = [np.binary_repr(x, width=width) for x in world_int]
    world_mask = np.array(list(reversed([list(reversed([bool(int(sym)) for sym in s])) for s in world_str])), dtype=bool)  # replacing with 1D tuple made code 5x slower
    mask_rows = len(world_int)
    mask_cols = width
    return world_mask, mask_rows, mask_cols


if __name__ == "__main__":
    # Locate the place for storing the frames
    if not os.path.exists("_frames"):
        os.mkdir("_frames")
    assert os.path.isdir("_frames"), \
        "'_frames' folder is used for storing the rendering results, but this name is already taken by some file"
    os.chdir("_frames")
    fname_prefix = "tmp"
    # Convert the world into a 2D numpy boolean-like array
    G, G_rows, G_cols = prepare_the_world(G_int)
    # Begin rendering
    print(f"This code will:\n"
          f"- render the rotating logo using ray casting,\n"
          f"- save the frames to '_frames' folder (in .ppm and .png formats),\n"
          f"- draw these frames on-screen")
    t0 = time.time()
    business_card_ray_tracing(G, fname_prefix, FRAMES)
    print(f"Tracing done in {time.time() - t0:.2f} seconds")
    show_frames(fname_prefix, FRAMES)
