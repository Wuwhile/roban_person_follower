import numpy as np


def decode_image(message):
    """Read ROS Image rows, including padding and the wire byte order."""
    encoding = message.encoding
    if encoding in ("rgb8", "bgr8"):
        dtype, channels = np.dtype("u1"), 3
    elif encoding == "16UC1":
        dtype, channels = np.dtype("u2"), 1
    elif encoding == "32FC1":
        dtype, channels = np.dtype("f4"), 1
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")

    height, width = message.height, message.width
    item_size = dtype.itemsize
    row_width = width * channels * item_size
    if height < 1 or width < 1 or message.step < row_width:
        raise ValueError("Invalid image dimensions or row stride")
    if len(message.data) != height * message.step:
        raise ValueError("Image data length does not match the header")

    wire_dtype = dtype.newbyteorder(">" if message.is_bigendian else "<")
    shape = (height, width)
    strides = (message.step, item_size * channels)
    if channels == 3:
        shape += (channels,)
        strides += (item_size,)
    pixels = np.ndarray(shape, dtype=wire_dtype, buffer=message.data, strides=strides)
    if encoding == "rgb8":
        pixels = pixels[:, :, ::-1]
    return np.ascontiguousarray(pixels, dtype=dtype)
