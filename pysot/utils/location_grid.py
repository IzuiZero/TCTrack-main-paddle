import paddle
def compute_locations(features,stride):
    h, w = features.shape[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride,
        features.place
    )
    return locations_per_level


def compute_locations_per_level(h, w, stride, place):
    shifts_x = paddle.arange(
        0, w * stride, step=stride,
        dtype='float32', place=place
    )
    shifts_y = paddle.arange(
        0, h * stride, step=stride,
        dtype='float32', place=place
    )
    shift_y, shift_x = paddle.meshgrid((shifts_y, shifts_x))
    shift_x = shift_x.reshape([-1])
    shift_y = shift_y.reshape([-1])
    # locations = paddle.stack((shift_x, shift_y), axis=1) + stride + 3*stride  # (size_z-1)/2*size_z 28
    # locations = paddle.stack((shift_x, shift_y), axis=1) + stride
    locations = paddle.stack((shift_x, shift_y), axis=1) + 32  #alex:48 // 32
    return locations
