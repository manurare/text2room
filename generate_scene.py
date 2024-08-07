import os
import json
from PIL import Image

from model.text2room_pipeline import Text2RoomPipeline
from model.utils.opt import get_default_parser
from model.utils.utils import save_poisson_mesh, generate_first_image

import torch
import numpy as np


def read_dpt(dpt_file_path):
    import numpy as np
    from struct import unpack
    """read depth map from *.dpt file.

    :param dpt_file_path: the dpt file path
    :type dpt_file_path: str
    :return: depth map data
    :rtype: numpy
    """
    TAG_FLOAT = 202021.25  # check for this when READING the file

    ext = os.path.splitext(dpt_file_path)[1]

    assert len(ext) > 0, ('readFlowFile: extension required in fname %s' % dpt_file_path)
    assert ext == '.dpt', exit('readFlowFile: fname %s should have extension ''.flo''' % dpt_file_path)

    fid = None
    try:
        fid = open(dpt_file_path, 'rb')
    except IOError:
        print('readFlowFile: could not open %s', dpt_file_path)

    tag = unpack('f', fid.read(4))[0]
    width = unpack('i', fid.read(4))[0]
    height = unpack('i', fid.read(4))[0]

    assert tag == TAG_FLOAT, ('readFlowFile(%s): wrong tag (possibly due to big-endian machine?)' % dpt_file_path)
    assert 0 < width and width < 100000, ('readFlowFile(%s): illegal width %d' % (dpt_file_path, width))
    assert 0 < height and height < 100000, ('readFlowFile(%s): illegal height %d' % (dpt_file_path, height))

    # arrange into matrix form
    depth_data = np.fromfile(fid, np.float32)
    depth_data = depth_data.reshape(height, width)

    fid.close()

    return depth_data


@torch.no_grad()
def main(args):
    # load trajectories
    trajectories = json.load(open(args.trajectory_file, "r"))

    # check if there is a custom prompt in the first trajectory
    # would use it to generate start image, if we have to
    if "prompt" in trajectories[0]:
        args.prompt = trajectories[0]["prompt"]

    # get first image from text prompt or saved image folder
    if (not args.input_image_path) or (not os.path.isfile(args.input_image_path)):
        first_image_pil = generate_first_image(args)
    else:
        first_depth = None
        if os.path.isfile(args.input_depth_path):
            ext = os.path.splitext(args.input_depth_path)[-1]
            if ext == ".dpt":
                first_depth = read_dpt(args.input_depth_path)
            elif ext == ".npy":
                first_depth = np.load(args.input_depth_path)
            else:
                raise "unknown depth format"
            first_depth[first_depth <= 0] = np.max(first_depth)

        first_image_pil = Image.open(args.input_image_path)

    # load pipeline
    pipeline = Text2RoomPipeline(args, first_image_pil=first_image_pil, first_depth=first_depth)

    # generate using all trajectories
    offset = 1  # have the start image already
    for t in trajectories:
        pipeline.set_trajectory(t)
        offset = pipeline.generate_images(offset=offset)

    # save outputs before completion
    pipeline.clean_mesh()
    intermediate_mesh_path = pipeline.save_mesh("after_generation.ply")
    save_poisson_mesh(intermediate_mesh_path, depth=args.poisson_depth, max_faces=args.max_faces_for_poisson)

    # run completion
    pipeline.args.update_mask_after_improvement = True
    pipeline.complete_mesh(offset=offset)
    pipeline.clean_mesh()

    # Now no longer need the models
    pipeline.remove_models()

    # save outputs after completion
    final_mesh_path = pipeline.save_mesh()

    # run poisson mesh reconstruction
    mesh_poisson_path = save_poisson_mesh(final_mesh_path, depth=args.poisson_depth, max_faces=args.max_faces_for_poisson)

    # save additional output
    pipeline.save_animations()
    pipeline.load_mesh(mesh_poisson_path)
    pipeline.save_seen_trajectory_renderings(apply_noise=False, add_to_nerf_images=True)
    pipeline.save_nerf_transforms()
    pipeline.save_seen_trajectory_renderings(apply_noise=True)

    print("Finished. Outputs stored in:", args.out_path)


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    main(args)