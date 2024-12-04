import numpy as np
from scipy.spatial.transform import Rotation as R

# import ipdb
import torch
from pytorch3d.transforms import Translate


def intersect_skew_line_groups(p, r, mask):
    # p, r both of shape (B, N, n_intersected_lines, 3)
    # mask of shape (B, N, n_intersected_lines)
    p_intersect, r = intersect_skew_lines_high_dim(p, r, mask=mask)
    if p_intersect is None:
        return None, None, None, None
    _, p_line_intersect = point_line_distance(
        p, r, p_intersect[..., None, :].expand_as(p)
    )
    intersect_dist_squared = ((p_line_intersect - p_intersect[..., None, :]) ** 2).sum(
        dim=-1
    )
    return p_intersect, p_line_intersect, intersect_dist_squared, r


def intersect_skew_lines_high_dim(p, r, mask=None):
    # Implements https://en.wikipedia.org/wiki/Skew_lines In more than two dimensions
    dim = p.shape[-1]
    # make sure the heading vectors are l2-normed
    if mask is None:
        mask = torch.ones_like(p[..., 0])
    r = torch.nn.functional.normalize(r, dim=-1)

    eye = torch.eye(dim, device=p.device, dtype=p.dtype)[None, None]
    I_min_cov = (eye - (r[..., None] * r[..., None, :])) * mask[..., None, None]
    sum_proj = I_min_cov.matmul(p[..., None]).sum(dim=-3)

    # I_eps = torch.zeros_like(I_min_cov.sum(dim=-3)) + 1e-10
    # p_intersect = torch.pinverse(I_min_cov.sum(dim=-3) + I_eps).matmul(sum_proj)[..., 0]
    p_intersect = torch.linalg.lstsq(I_min_cov.sum(dim=-3), sum_proj).solution[..., 0]

    # I_min_cov.sum(dim=-3): torch.Size([1, 1, 3, 3])
    # sum_proj: torch.Size([1, 1, 3, 1])

    # p_intersect = np.linalg.lstsq(I_min_cov.sum(dim=-3).numpy(), sum_proj.numpy(), rcond=None)[0]

    if torch.any(torch.isnan(p_intersect)):
        print(p_intersect)
        return None, None
        ipdb.set_trace()
        assert False
    return p_intersect, r


def point_line_distance(p1, r1, p2):
    df = p2 - p1
    proj_vector = df - ((df * r1).sum(dim=-1, keepdim=True) * r1)
    line_pt_nearest = p2 - proj_vector
    d = (proj_vector).norm(dim=-1)
    return d, line_pt_nearest


def compute_optical_axis_intersection(cameras, in_ndc=True):
    centers = cameras.get_camera_center()
    principal_points = cameras.principal_point

    one_vec = torch.ones((len(cameras), 1), device=centers.device)
    optical_axis = torch.cat((principal_points, one_vec), -1)

    # optical_axis = torch.cat(
    #     (principal_points, cameras.focal_length[:, 0].unsqueeze(1)), -1
    # )

    pp = cameras.unproject_points(optical_axis, from_ndc=in_ndc, world_coordinates=True)
    pp2 = torch.diagonal(pp, dim1=0, dim2=1).T

    directions = pp2 - centers
    centers = centers.unsqueeze(0).unsqueeze(0)
    directions = directions.unsqueeze(0).unsqueeze(0)

    p_intersect, p_line_intersect, _, r = intersect_skew_line_groups(
        p=centers, r=directions, mask=None
    )

    if p_intersect is None:
        dist = None
    else:
        p_intersect = p_intersect.squeeze().unsqueeze(0)
        dist = (p_intersect - centers).norm(dim=-1)

    return p_intersect, dist, p_line_intersect, pp2, r


def normalize_cameras_with_up_axis(cameras, sequence_name, scale=1.0, in_ndc=True):
    """
    Normalizes cameras such that the optical axes point to the origin and the average
    distance to the origin is 1.

    Args:
        cameras (List[camera]).
    """

    # Let distance from first camera to origin be unit
    new_cameras = cameras.clone()
    new_transform = new_cameras.get_world_to_view_transform()

    p_intersect, dist, p_line_intersect, pp, r = compute_optical_axis_intersection(
        cameras,
        in_ndc=in_ndc
    )
    t = Translate(p_intersect)

    # scale = dist.squeeze()[0]
    scale = dist.squeeze().mean()

    # Degenerate case
    if scale == 0:
        print(cameras.T)
        print(new_transform.get_matrix()[:, 3, :3])
        return -1
    assert scale != 0

    new_transform = t.compose(new_transform)
    new_cameras.R = new_transform.get_matrix()[:, :3, :3]
    new_cameras.T = new_transform.get_matrix()[:, 3, :3] / scale * 1.85

    needs_checking = False

    # ===== Rotation normalization
    # Estimate the world 'up' direction assuming that yaw is small
    # and running SVD on the x-vectors of the cameras
    x_vectors = new_cameras.R.transpose(1, 2)[:, 0, :].clone()
    x_vectors -= x_vectors.mean(dim=0, keepdim=True)
    U, S, Vh = torch.linalg.svd(x_vectors)
    V = Vh.mH
    # vector with the smallest variation is to the normal to
    # the plane of x-vectors (assume this to be the up direction)
    if S[0] / S[1] > S[1] / S[2]:
        print('Warning: unexpected singular values in sequence {}: {}'.format(sequence_name, S))
        needs_checking = True
        # return None, None, None, None, None
    estimated_world_up = V[:, 2:]
    # check all cameras have the same y-direction
    for camera_idx in range(len(new_cameras.T)):
        if torch.sign(torch.dot(estimated_world_up[:, 0],
                                new_cameras.R[0].transpose(0,1)[1, :])) != torch.sign(torch.dot(estimated_world_up[:, 0],
                                    new_cameras.R[camera_idx].transpose(0,1)[1, :])):
            print("Some cameras appear to be flipped in sequence {}".format(sequence_name) )
            needs_checking = True
            # return None, None, None, None, None
    flip = torch.sign(torch.dot(estimated_world_up[:, 0], new_cameras.R[0].transpose(0,1)[1, :])) < 0
    if flip:
        estimated_world_up = V[:, 2:] * -1
    # build the target coordinate basis using the estimated world up
    target_coordinate_basis = torch.cat([V[:, :1],
                                        estimated_world_up,
                                        torch.linalg.cross(V[:, :1], estimated_world_up, dim=0)],
                                        dim=1)
    new_cameras.R = torch.matmul(target_coordinate_basis.T, new_cameras.R)
    return new_cameras, p_intersect, p_line_intersect, pp, r, needs_checking


def dot(x, y):
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x, eps=1e-20):
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x, eps=1e-20):
    return x / length(x, eps)


def look_at(campos, target, opengl=True):
    # campos: [N, 3], camera/eye position
    # target: [N, 3], object to look at
    # return: [N, 3, 3], rotation matrix
    if not opengl:
        # camera forward aligns with -z
        forward_vector = safe_normalize(target - campos)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(forward_vector, up_vector))
        up_vector = safe_normalize(np.cross(right_vector, forward_vector))
    else:
        # camera forward aligns with +z
        forward_vector = safe_normalize(campos - target)
        up_vector = np.array([0, 1, 0], dtype=np.float32)
        right_vector = safe_normalize(np.cross(up_vector, forward_vector))
        up_vector = safe_normalize(np.cross(forward_vector, right_vector))
    R = np.stack([right_vector, up_vector, forward_vector], axis=1)
    return R


# elevation & azimuth to pose (cam2world) matrix
def orbit_camera(elevation, azimuth, radius=1, is_degree=True, target=None, opengl=True):
    # radius: scalar
    # elevation: scalar, in (-90, 90), from +y to -y is (-90, 90)
    # azimuth: scalar, in (-180, 180), from +z to +x is (0, 90)
    # return: [4, 4], camera pose matrix
    if is_degree:
        elevation = np.deg2rad(elevation)
        azimuth = np.deg2rad(azimuth)
    x = radius * np.cos(elevation) * np.sin(azimuth)
    y = - radius * np.sin(elevation)
    z = radius * np.cos(elevation) * np.cos(azimuth)
    if target is None:
        target = np.zeros([3], dtype=np.float32)
    campos = np.array([x, y, z]) + target  # [3]
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = look_at(campos, target, opengl)
    T[:3, 3] = campos

    return T


def mat2latlon(T):
    if not isinstance(T, np.ndarray):
        xyz = T.cpu().detach().numpy()
    else:
        xyz = T.copy()
    r = np.linalg.norm(xyz)
    xyz = xyz / r
    theta = -np.arcsin(xyz[1])
    azi = np.arctan2(xyz[0], xyz[2])
    return np.rad2deg(theta), np.rad2deg(azi), r


def extract_camera_properties(camera_to_world_matrix):
    # Camera position is the translation part of the matrix
    camera_position = camera_to_world_matrix[:3, 3]

    # Extracting the forward direction vector (third column of rotation matrix)
    forward = camera_to_world_matrix[:3, 2]

    return camera_position, forward


def compute_angular_error_batch(rotation1, rotation2):
    R_rel = np.einsum("Bij,Bjk ->Bik", rotation1.transpose(0, 2, 1), rotation2)
    t = (np.trace(R_rel, axis1=1, axis2=2) - 1) / 2
    theta = np.arccos(np.clip(t, -1, 1))
    return theta * 180 / np.pi


def find_mask_center_and_translate(image, mask):
    """
    Calculate the center of the mask and translate the image such that
    the mask center is at the image center.

    Args:
    - image (torch.Tensor): Input image tensor of shape (N, C, H, W)
    - mask (torch.Tensor): Mask tensor of shape (N, 1, H, W)

    Returns:
    - Translated image of shape (N, C, H, W)
    """
    _, _, h, w = image.shape

    # Calculate the center of mass of the mask
    # Note: mask should be a binary mask of the same spatial dimensions as the image
    y_coords, x_coords = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    total_mass = mask.sum(dim=[2, 3], keepdim=True)
    x_center = (mask * x_coords.to(image.device)).sum(dim=[2, 3], keepdim=True) / total_mass
    y_center = (mask * y_coords.to(image.device)).sum(dim=[2, 3], keepdim=True) / total_mass

    # Calculate the translation needed to move the mask center to the image center
    image_center_x, image_center_y = w // 2, h // 2
    delta_x = x_center.squeeze() - image_center_x
    delta_y = y_center.squeeze() - image_center_y

    return torch.tensor([delta_x, delta_y])


def create_voxel_grid(length, resolution=64):
    """
    Creates a voxel grid.
    xyz_range: ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    resolution: The number of divisions along each axis.
    Returns a 4D tensor representing the voxel grid, with each voxel initialized to 1 (solid).
    """
    x = torch.linspace(-length, length, resolution)
    y = torch.linspace(-length, length, resolution)
    z = torch.linspace(-length, length, resolution)
    
    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
    voxels = torch.stack([xx, yy, zz, torch.ones_like(xx)], dim=-1)  # Homogeneous coordinates
    return voxels


def project_voxels_to_image(voxels, camera):
    """
    Projects voxel centers into the camera's image plane.
    voxels: 4D tensor of voxel grid in homogeneous coordinates.
    K: Camera intrinsic matrix.
    R: Camera rotation matrix.
    t: Camera translation vector.
    Returns a tensor of projected 2D points in image coordinates.
    """
    device = voxels.device
    # K, R, t = torch.tensor(K, device=device), torch.tensor(R, device=device), torch.tensor(t, device=device)

    # Flatten voxels to shape (N, 4) for matrix multiplication
    N = voxels.nelement() // 4  # Total number of voxels
    voxels_flat = voxels.reshape(-1, 4).t()  # Shape (4, N)

    # # Apply extrinsic parameters (rotation and translation)
    # transformed_voxels = R @ voxels_flat[:3, :] + t[:, None]

    # # Apply intrinsic parameters
    # projected_voxels = K @ transformed_voxels

    projected_voxels = camera.projection_matrix.transpose(0, 1) @ camera.world_view_transform.transpose(0, 1) @ voxels_flat    

    # Convert from homogeneous coordinates to 2D
    projected_voxels_2d = (projected_voxels[:2, :] / projected_voxels[3, :]).t() # Reshape to grid dimensions with 2D points
    projected_voxels_2d = (projected_voxels_2d.reshape(*voxels.shape[:-1], 2) + 1.) * 255 * 0.5

    return projected_voxels_2d


def carve_voxels(voxel_grid, projected_points, mask):
    """
    Updates the voxel grid based on the comparison with the mask.
    voxel_grid: 3D tensor representing the voxel grid.
    projected_points: Projected 2D points in image coordinates.
    mask: Binary mask image.
    """
    # Convert projected points to indices in the mask
    indices_x = torch.clamp(projected_points[..., 0], 0, mask.shape[1] - 1).long()
    indices_y = torch.clamp(projected_points[..., 1], 0, mask.shape[0] - 1).long()

    # Check if projected points are within the object in the mask
    in_object = mask[indices_y, indices_x]

    # Carve out voxels where the projection does not fall within the object
    voxel_grid[in_object == 0] = 0


def sample_points_from_voxel(cameras, masks, length=1, resolution=64, N=5000, inverse=False, device="cuda"):
    """
    Randomly sample N points from solid regions in a voxel grid.
    
    Args:
    - voxel_grid (torch.Tensor): A 3D tensor representing the voxel grid after carving.
      Solid regions are marked with 1s.
    - N (int): The number of points to sample.
    
    Returns:
    - sampled_points (torch.Tensor): A tensor of shape (N, 3) representing the sampled 3D coordinates.
    """
    voxel_grid = create_voxel_grid(length, resolution).to(device)
    voxel_grid_indicator = torch.ones(resolution, resolution, resolution)

    masks = torch.from_numpy(masks).to(device).squeeze()

    for idx, cam in enumerate(cameras):
        projected_points = project_voxels_to_image(voxel_grid, cam)
        carve_voxels(voxel_grid_indicator, projected_points, masks[idx])

    voxel_grid_indicator = voxel_grid_indicator.reshape(resolution, resolution, resolution)

    # Identify the indices of solid voxels
    if inverse:
        solid_indices = torch.nonzero(voxel_grid_indicator == 0)
    else:
        solid_indices = torch.nonzero(voxel_grid_indicator == 1)
    
    # Randomly select N indices from the solid indices
    if N <= solid_indices.size(0):
        # Randomly select N indices from the solid indices if there are enough solid voxels
        sampled_indices = solid_indices[torch.randperm(solid_indices.size(0))[:N]]
    else:
        # If there are not enough solid voxels, sample with replacement
        sampled_indices = solid_indices[torch.randint(0, solid_indices.size(0), (N,))]
    
    # Convert indices to coordinates
    # Note: This step assumes the voxel grid spans from 0 to 1 in each dimension.
    # Adjust accordingly if your grid spans a different range.
    sampled_points = sampled_indices.float() / (voxel_grid.size(0) - 1) * 2 * length - length
    
    return sampled_points


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60, near=0.01, far=100):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.fovy = np.deg2rad(fovy)  # deg 2 rad
        self.near = near
        self.far = far
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_matrix(np.eye(3))
        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!

    @property
    def fovx(self):
        return 2 * np.arctan(np.tan(self.fovy / 2) * self.W / self.H)

    @property
    def campos(self):
        return self.pose[:3, 3]

    # pose (c2w)
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] = self.radius  # opengl convention...
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    # view (w2c)
    @property
    def view(self):
        return np.linalg.inv(self.pose)

    # projection (perspective)
    @property
    def perspective(self):
        y = np.tan(self.fovy / 2)
        aspect = self.W / self.H
        return np.array(
            [
                [1 / (y * aspect), 0, 0, 0],
                [0, -1 / y, 0, 0],
                [
                    0,
                    0,
                    -(self.far + self.near) / (self.far - self.near),
                    -(2 * self.far * self.near) / (self.far - self.near),
                ],
                [0, 0, -1, 0],
            ],
            dtype=np.float32,
        )

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(self.fovy / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2], dtype=np.float32)

    @property
    def mvp(self):
        return self.perspective @ np.linalg.inv(self.pose)  # [4, 4]

    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0]
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([-dx, -dy, dz])